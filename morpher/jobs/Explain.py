import json
import os.path
import traceback
from collections import defaultdict

import pandas as pd
import jsonpickle as jp
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from morpher.jobs import MorpherJob, Retrieve
from morpher.exceptions import kwargs_not_empty


class Explain(MorpherJob):
    def do_execute(self):

        # experiment_mode 2 is an interpretation experiment, running different interpretation algorithm
        experiment_mode = 2

        # if we have a list of filenames coming from 'Split', we pass over 'train' and 'test' respectively;
        # otherwise we pass over the file we got, which by default is the one file attached to the cohort which generated the model

        if type(self.get_input("filenames")) == list:
            train, test = self.get_input("filenames")
        else:
            task = self.get_task()
            users_path = os.path.abspath(
                self.config.get("paths", "user_files")
            )
            filename = task["parameters"]["file"]["name"]
            filename = os.path.join(users_path, filename)
            train, test = f"{filename}_train", f"{filename}_test"

        train = pd.read_csv(filepath_or_buffer=train)
        test = pd.read_csv(filepath_or_buffer=test)

        model_ids = self.get_input("model_ids")
        response = [
            (jp.decode(json.dumps(model["content"])), model["parameters"])
            for model in Retrieve(self.session).get_models(model_ids)
        ]
        models = [model[0] for model in response]
        features = [model[1]["features"] for model in response]
        models_features = dict(
            zip(
                [model.__class__.__name__ for model in models],
                [feat for feat in features]
            )
        )
        
        # go for zip here, model_id_mapping
        model_id_mapping = dict(
            zip([model.__class__.__name__ for model in models], model_ids)
        )

        cohort_id = self.get_input("cohort_id")
        user_id = self.get_input("user_id")
        target = self.get_input("target")
        explainers = self.get_input_variables("explainers")
        
        #make it become a list if not already
        assert explainers != ""
        if type(explainers) is str:
            explainers = [explainers]

        explanations = self.execute(
            train,
            target=target,
            models={model.__class__.__name__: model for model in models},
            explainers=explainers,
            models_features=models_features, 
            exp_kwargs={"test": test}
        )

        for model in models:
            clf_name = model.__class__.__name__
            model_id = model_id_mapping[clf_name]
            description = "Explanations for target '{target}' based on {methods}".format(
                target=target, methods=", ".join(explainers)
            )

            #remove type reference to allow deserialization
            explanations[clf_name] = {
                exp_name.__class__.__name__ if callable(exp_name) \
                else exp_name : explanations[clf_name][exp_name] \
                for exp_name in explanations[clf_name]
            }

            self.add_experiment(
                cohort_id=cohort_id,
                model_id=model_id,
                user_id=user_id,
                description=description,
                target=target,
                experiment_mode=experiment_mode,
                parameters=explanations[clf_name],
            )

        self.logger.info("Models explained successfully.")

    def add_experiment(self, **kwargs):

        response = self.api("experiments", "new", data=kwargs)

        if response.get("status") == "success":
            return response.get("experiment_id")
        else:
            raise Exception(
                "There was an error creating an Experiment. Server returned: %s"
                % response.get("msg")
            )

    def execute(self, data, **kwargs):

        models = kwargs.get("models")
        explainers = kwargs.get("explainers")
        target = kwargs.get("target")
        kwargs_not_empty(models, "models")
        kwargs_not_empty(explainers, "explainers")
        kwargs_not_empty(target, "target")
        models_features = kwargs.get("models_features") or {}

        exp_kwargs = kwargs.get("exp_kwargs") or {}
        test = exp_kwargs.get("test")
        if not isinstance(test, pd.DataFrame):
            test = pd.DataFrame()

        try:
            if not data.empty and models and target and explainers:
                
                explanations = defaultdict(lambda: {})
                for clf_name in models:
        
                    # include zero-out features, in case not all are available
                    # get the features in the correct order that model expects them
                    feats = models_features.get(clf_name)

                    if feats:
                        # include target, because explain job needs it
                        feats = [target] + feats

                        for feat in feats:
                            if feat not in list(data.columns):
                                data[feat] = 0.0
                            if not test.empty:
                                if feat not in list(test.columns):
                                    test[feat] = 0.0
                        data = data[feats]
                        if not test.empty:
                            exp_kwargs["test"] = test[feats]

                    model = models[clf_name]
                    for exp_name in explainers:

                        if not callable(exp_name):
                            exp_name = self.get_callable('morpher.explainers', exp_name)

                        explainer = exp_name(
                            data,
                            model,
                            target,
                            **exp_kwargs
                        ) #instantiate the algorithm in runtime
                        explanations[clf_name][exp_name] = explainer.explain(
                            **exp_kwargs
                        )

                return explanations

            else:
                raise AttributeError(
                    "No data provided, models or target not available"
                )
        except Exception:
            self.logger.error(traceback.format_exc())
            return None
