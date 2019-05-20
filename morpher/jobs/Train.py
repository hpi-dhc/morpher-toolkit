#!/usr/bin/env python
import traceback
import logging
from morpher.exceptions import kwarg_not_empty
from morpher.algorithms import *
import morpher.config as config
from morpher.jobs import MorpherJob
import pandas as pd
import numpy as np
import os.path
import pandas as pd
import json
import jsonpickle as jp

class Train(MorpherJob):

    def do_execute(self):
        
        #if we have a list of filenames coming from 'Split', we pass over the 'train' set for training; otherwise we pass over the file we got
        filename = self.get_input("filename") or self.get_input("filenames")[0]
        df = pd.read_csv(filepath_or_buffer= filename)
        algorithms = self.get_input_variables("algorithms")
        
        assert algorithms != ""
        if type(algorithms) is str: #make it become a list if not already
            algorithms = [algorithms]
        
        target = self.get_input_variables("target")
        models = self.execute(df, target=target, algorithms=algorithms)

        params = {}
        params["target"] = target
        params["features"] = list(df.drop(target, axis=1).columns)
        
        #store each model in the database and generate a dict in the form {"DecisionTree":999}
        model_ids = {}
        for model in models:
            model_ids[model.__class__.__name__] = self.add(model, params)

        #if we have a list of filenames coming from 'Split', we pass over the filenames
        if type(self.get_input("filenames")) == list:
            self.add_output("filenames", self.get_input("filenames"))
        else:
            self.add_output("filename", filename)
        
        self.add_output("target", target)
        self.add_output("model_ids", model_ids)
        self.add_output("cohort_id", 9999)
        self.logger.info("Models trained successfully.")

    def add(self, model, params):
        
        params["hyperparameters"] = model.get_params()
        data = {}
        data["task_id"] = self.task_id
        data["name"] = model.__class__.__name__
        data["fqn"] =  model.clf.__class__.__module__ + '.' + model.clf.__class__.__qualname__        
        data["content"] = json.loads(jp.encode(model))
        data["parameters"] = params

        response = self.api("models", "new", data)

        if response.get("status") == "error":
            raise Exception("Error inserting new model. Check the server")

        return response.get("model_id")

    def execute(self, data, target, **kwargs):
        try:

          if not data.empty:

            ''' if split_data was called beforehand, data contains a subset of the original available data '''
            labels = data[target]
            features = data.drop(target, axis=1)
            algorithms = kwargs.get("algorithms")
            kwarg_not_empty(algorithms,"algorithms")

            trained_models = []

            for algorithm in algorithms:
                clf = eval("{algorithm}()".format(algorithm=algorithm)) #instantiate the algorithm in runtime

                clf.fit(features, labels)
                trained_models.append(clf)

            return trained_models

          else:
            raise AttributeError("No data provided")        

        except Exception as e:
          logging.error(traceback.format_exc())

        return data


