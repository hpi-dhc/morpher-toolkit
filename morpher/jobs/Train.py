import json
import logging
import traceback

import jsonpickle as jp
import pandas as pd

from morpher.exceptions import kwargs_not_empty
from morpher.jobs import MorpherJob

class Train(MorpherJob):
    def do_execute(self):

        # if we have a list of filenames coming from 'Split', we pass over the 'train' set for training; otherwise we pass over the file we got
        filename = self.get_input("filename") or self.get_input("filenames")[0]
        task = self.get_task()
        cohort_params = task["parameters"].get("cohort_params")
        drop = [] # cohort features to be removed are stored as a 'remove' list
        if cohort_params:
            drop = cohort_params.get("remove") or []

        df = pd.read_csv(filepath_or_buffer=filename)
        algorithms = self.get_input_variables("algorithms")

        assert algorithms != ""
        if type(algorithms) is str:  # make it become a list if not already
            algorithms = [algorithms]

        target = self.get_input_variables("target")
        models = self.execute(df, target=target, algorithms=algorithms, drop=drop)

        params = {}
        params["cohort_id"] = self.get_input("cohort_id")
        params["user_id"] = self.get_input("user_id")
        params["target"] = target
        params["features"] = [feat for feat in df.drop(target, axis=1).columns if feat not in drop]

        # store each model in the database and generate a dict in the form {"DecisionTree":999}
        model_ids = self.persist(models, params)

        # if we have a list of filenames coming from 'Split', we pass over the filenames
        if type(self.get_input("filenames")) == list:
            self.add_output("filenames", self.get_input("filenames"))
        else:
            self.add_output("filename", filename)

        self.add_output("target", target)
        self.add_output("model_ids", model_ids)
        self.add_output("cohort_id", params["cohort_id"])
        self.add_output("user_id", params["user_id"])
        self.logger.info("Models trained successfully.")

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
        data["name"] = model.__class__.__name__ + " for " + params["target"]
        data["fqn"] = (
            model.clf.__class__.__module__
            + "."
            + model.clf.__class__.__qualname__
        )
        data["content"] = json.loads(jp.encode(model))
        data["parameters"] = params

        response = self.api("models", "new", data)

        if response.get("status") == "error":
            raise Exception(
                "Error inserting new model. Check the server. Message returned: {msg}".format(
                    msg=response.get("msg")
                )
            )

        return response.get("model_id")

    def execute(self, data, target, **kwargs):
        try:

            if not data.empty:

                """ if split_data was called beforehand, data contains a subset of the original available data """
                labels = data[target]
                features = data.drop(target, axis=1)
                params = {}
                algorithms = kwargs.get("algorithms")
                kwargs_not_empty(algorithms, "algorithms")
                hyperparams = kwargs.get("hyperparams")
                optimize = kwargs.get("optimize")
                param_grid = kwargs.get("param_grid")
                crossval = kwargs.get("crossval")
                n_splits = kwargs.get("n_splits")
                drop = kwargs.get("drop")  # list of features to drop
                verbose = kwargs.get("verbose")  # verbose mode

                trained_models = {}
                crossval_metrics = {}

                if drop:
                    cols_to_drop = [col for col in drop if col in features.columns]
                    features = features.drop(cols_to_drop, axis=1)

                for algorithm in algorithms:

                    """ here for compatibility purposes """
                    if not callable(algorithm):
                        algorithm = self.get_callable('morpher.algorithms', algorithm)

                    clf = algorithm(
                        hyperparams=hyperparams,
                        optimize=optimize,
                        param_grid=param_grid,
                        crossval=crossval,
                        n_splits=n_splits,
                        verbose=verbose,

                    )  # instantiate the algorithm in runtime

                    """ if fit returns anything, it will be the cross_validated metrics """
                    if crossval:
                        crossval_metrics[algorithm] = clf.fit(features, labels)
                    else:
                        clf.fit(features, labels)

                    trained_models[algorithm] = clf

                if kwargs.get("persist") is True:
                    params["target"] = target
                    params["features"] = features.columns
                    self.persist(trained_models, params)

                if crossval:
                    return trained_models, crossval_metrics

                return trained_models

            else:
                raise AttributeError("No data provided")

        except Exception:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return data