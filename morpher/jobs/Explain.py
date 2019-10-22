#!/usr/bin/env python
import traceback
import logging
from morpher.jobs import MorpherJob
from morpher.jobs import Retrieve
from morpher.exceptions import kwarg_not_empty
from morpher.algorithms import *
from morpher.explainers import *
from morpher.metrics import *
import os.path
import pandas as pd
import json
import jsonpickle as jp
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error


class Explain(MorpherJob):

    def do_execute(self):
        
        #if we have a list of filenames coming from 'Split', we pass over the 'test' set for evaluation (pos. 1); otherwise we pass over the file we got
        if type(self.get_input("filenames")) == list:
            filename = self.get_input("filenames")[1]
            validation_mode = 0            
        else:
            filename = self.get_input("filename")
            validation_mode = 1

        df = pd.read_csv(filepath_or_buffer= filename)

        model_ids = self.get_input("model_ids")  
        models = [jp.decode(json.dumps(model["content"])) for model in Retrieve(self.session).get_models(model_ids)]

        #go for zip here, model_id_mapping
        model_id_mapping = dict(zip([model.__class__.__name__ for model in models], model_ids))

        cohort_id = self.get_input("cohort_id")
        user_id = self.get_input("user_id")
        target = self.get_input("target")

        results = self.execute(df, target=target, models={model.__class__.__name__: model for model in models})

        for model in models:
            clf_name = model.__class__.__name__
            model_id = model_id_mapping[clf_name]
            description = "Model based on {clf_name} for target '{target}'".format(clf_name=clf_name, target=target)
            predictions = [ { "target_label": float(results[clf_name]["y_true"].iloc[i]),"predicted_label": float(results[clf_name]["y_pred"][i]),"predicted_proba": float(results[clf_name]["y_probs"][i]) } for i in range(len(results[clf_name]["y_true"])) ]
            disc_metrics = get_discrimination_metrics(results[clf_name]["y_true"], results[clf_name]["y_pred"], results[clf_name]["y_probs"])
            cal_metrics = get_calibration_metrics(results[clf_name]["y_true"], results[clf_name]["y_probs"])
            cu_metrics = get_clinical_usefulness_metrics(disc_metrics)
            experiment_id = self.add_experiment(cohort_id=cohort_id, model_id=model_id,user_id=user_id,description=description,target=target,validation_mode=validation_mode,parameters={"discrimination": disc_metrics, "calibration": cal_metrics, "clinical_usefulness": cu_metrics})
            self.add_batch(experiment_id, predictions)

        self.logger.info("Algorithms evaluated successfully.")

    def execute(self, data, **kwargs):

        models = kwargs.get("models")
        explainers = kwargs.get("explainers")
        target = kwargs.get("target")
        kwarg_not_empty(models,"models")
        kwarg_not_empty(explainers,"explainers")
        kwarg_not_empty(target,"target")
        exp_kwargs = kwargs.get("exp_kwargs") or {}

        try:
            if not data.empty and models and target and explainers:
                
                explanations = defaultdict(lambda: {})                
                for model_name in models:
                    model = models[model_name]
                    for exp_name in explainers:                     
                        explainer = globals()[exp_name](data, model, target, **exp_kwargs) #instantiate the algorithm in runtime
                        explanations[model_name][exp_name] = explainer.explain(**exp_kwargs)

                return explanations

            else:
                raise AttributeError("No data provided, models or target not available")
        except Exception as e:
            self.logger.error(traceback.format_exc())
            return None

    def print_clf_performance(self, clf_name, y_true, y_pred, y_probs):
        '''
        Prints performance of the prediction results
        '''
        print("***Performance report for {}".format(clf_name))

        ''' report predictions '''
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Classification report:")
        print(classification_report(y_true, y_pred))
        print("AUROC score:")
        print(roc_auc_score(y_true, y_probs))
        print("DOR:")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        dor = (tp/fp)/(fn/tn)
        print(dor)
        print("***\n")




