import traceback
import json

import jsonpickle as jp
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from morpher.jobs import MorpherJob
from morpher.jobs import Retrieve


class Calibrate(MorpherJob):

    def do_execute(self):

        # if we have a list of filenames coming from 'Split', we pass over the 'test' set for evaluation (pos. 1); otherwise we pass over the file we got
        if type(self.get_input("filenames")) == list:
            filename = self.get_input("filenames")[1]
        else:
            filename = self.get_input("filename")

        df = pd.read_csv(filepath_or_buffer=filename)

        cohort_id = self.get_input("cohort_id")
        user_id = self.get_input("user_id")
        target = self.get_input("target")
        model_ids = self.get_input("model_ids")
        calibration_method = self.get_input_variables("calibration_method")

        params = {}
        params["cohort_id"] = cohort_id
        params["user_id"] = user_id
        params["target"] = target
        params["calibration_method"] = calibration_method
        params["features"] = list(df.drop(target, axis=1).columns)

        models = [jp.decode(json.dumps(model["content"])) for model in Retrieve(self.session).get_models(model_ids)]
        calibrated_models = self.execute(df, target=target, models={model.__class__.__name__: model for model in models}, method=calibration_method)

        # store each model in the database and generate a dict in the form {"DecisionTree":999}
        calibrated_model_ids = self.persist(calibrated_models, params)

        self.add_output("target", target)
        self.add_output("model_ids", calibrated_model_ids)
        self.add_output("cohort_id", params["cohort_id"])
        self.add_output("user_id", params["user_id"])
        self.logger.info("Algorithms calibrated successfully.")

    def persist(self, models, params):
        model_ids = []
        for clf_name in models:
            model_ids.append(self.add(models[clf_name], params))

        return model_ids

    def add(self, model, params):

        params["hyperparameters"] = model.get_params()
        data = {}
        data["task_id"] = self.task_id
        data["cohort_id"] = params["cohort_id"]
        data["user_id"] = params["user_id"]
        data["name"] = model.__class__.__name__ + " for " + params["target"] + " with Calibration (" + params['calibration_method'] + ")"
        data["fqn"] = model.__class__.__module__ + '.' + model.__class__.__qualname__
        data["content"] = json.loads(jp.encode(model))
        data["parameters"] = params

        response = self.api("models", "new", data)

        if response.get("status") == "error":
            raise Exception("Error inserting new model. Check the server. Message returned: {msg}".format(msg=response.get("msg")))

        return response.get("model_id")

    def execute(self, data, target, models, method='isotonic', **kwargs):

        calibrated_models = {}

        try:
            if not data.empty and models and target:
                y_train = data[target]
                X_train = data.drop(target, axis=1)

                for clf_name in models:
                    clf = models[clf_name].clf
                    print(f"Performing calibration for {clf_name}")
                    calibrated_clf = CalibratedClassifierCV(clf, cv='prefit', method=method)
                    calibrated_clf.fit(X_train, y_train)
                    calibrated_models[clf_name] = calibrated_clf
                return calibrated_models
            else:
                raise AttributeError("No data provided, algorithms or target not available")
        except Exception as e:
            print(e)
            self.logger.error(traceback.format_exc())
        return None
