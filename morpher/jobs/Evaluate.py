#!/usr/bin/env python
import traceback
import logging
from morpher.jobs import MorpherJob
from morpher.jobs import Retrieve
from morpher.exceptions import kwarg_not_empty
from morpher.algorithms import *
from morpher.metrics import *
import os.path
import pandas as pd
import json
import jsonpickle as jp
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error


class Evaluate(MorpherJob):

    def do_execute(self):
        
        #if we have a list of filenames coming from 'Split', we pass over the 'test' set for evaluation (pos. 1); otherwise we pass over the file we got
        if type(self.get_input("filenames")) == list:
            filename = self.get_input("filenames")[1]
            experiment_mode = 0            
        else:
            filename = self.get_input("filename")
            experiment_mode = 1

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
            experiment_id = self.add_experiment(cohort_id=cohort_id, model_id=model_id,user_id=user_id,description=description,target=target,experiment_mode=experiment_mode,parameters={"discrimination": disc_metrics, "calibration": cal_metrics, "clinical_usefulness": cu_metrics})
            self.add_batch(experiment_id, predictions)

        self.logger.info("Algorithms evaluated successfully.")

    def add_batch(self, experiment_id, predictions):

        response = self.api("experiments/predictions/{0}".format(experiment_id), "batch", data=predictions)
        return response.get('status')

    def add_experiment(self, **kwargs):
     
        response = self.api("experiments", "new", data=kwargs)

        if response.get("status") == "success":
            return response.get("experiment_id")
        else:
            raise Exception("There was an error creating an Experiment. Server returned: %s" % response.get("msg"))

    def execute(self, data, target, models, print_performance=False, **kwargs):
        try:
            if not data.empty and models and target:
                results = {}
                labels = data[target] #true labels
                features = data.drop(target, axis=1)
                for clf_name in models:
                    clf = models[clf_name]
                    y_true, y_pred, y_probs = labels, clf.predict(features), clf.predict_proba(features)[:,1]
                    results[clf_name] = { "y_true": y_true, "y_pred": y_pred, "y_probs": y_probs}
                    if print_performance:
                        self.print_clf_performance(clf_name, y_true, y_pred, y_probs)
                return results
            else:
                raise AttributeError("No data provided, algorithms or target not available")
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




