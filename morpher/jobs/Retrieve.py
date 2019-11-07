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

class Retrieve(MorpherJob):

    def do_execute(self):
        
        filename = self.get_input("filename")
        df = pd.read_csv(filepath_or_buffer= filename)
        model_ids = self.get_input_variables("models")

        task = self.get_task()
        
        assert model_ids != ""
        if type(model_ids) is str: #make it become a list if not already
            model_ids = [model_ids]
        
        target = self.get_input_variables("target")

        models = self.get_models(model_ids, details=True)

        params = {}
        params["cohort_id"] = self.get_input("cohort_id") or task["parameters"]["cohort_id"]
        params["user_id"] = self.get_input("user_id") or task["parameters"]["user_id"]
        params["target"] = target        
        params["features"] = list(df.drop(target, axis=1).columns)
        
        #validate now if the models at hand match the information provided via the arguments/parameters
        error_summary = ""
        model_ids = [] # if the models are validated, we pass over its ID

        try:

            for model in models:

                error_msg = ""
                if model["parameters"]["target"] != target:
                    error_msg += "Targets do not match for model '{0}'. Expected '{1}', but '{2}' was provided. \n".format(model["name"], model["target"], target)

                if model["parameters"]["features"] != params["features"]:
                    error_msg += "Features do not match for model '{0}'. Please check data provided. \n".format(model["name"])
                
                if error_msg == "":
                    model_ids.append(model["id"])

                error_summary += error_msg

            if error_summary != "":
                raise AttributeError("Error while validating models and inputs. Error summary: {0}".format(error_summary))

        except Exception as e:            
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        #if we have a list of filenames coming from previous job, we pass it over
        if type(self.get_input("filenames")) == list:
            self.add_output("filenames", self.get_input("filenames"))
        else:
            self.add_output("filename", filename)
        
        self.add_output("target", target)
        self.add_output("model_ids", model_ids)
        self.add_output("cohort_id", params["cohort_id"])
        self.add_output("user_id", params["user_id"])
        self.logger.info("Models retrieved successfully.")

    def get_models(self, model_ids, details=False):
        
        if details:
            data = {"model_ids": model_ids, "details": "yes"}
        else:
            data = {"model_ids": model_ids}
        
        response = self.api("models", "get", data=data)

        if response.get("status") == "success":
            return response.get("models")
        else:
            raise Exception("There was an error retrieving the trained models. Check the server")
