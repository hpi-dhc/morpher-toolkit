#!/usr/bin/env python
import traceback
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error
from collections import defaultdict

def get_discrimination_metrics(y_true, y_pred, y_probs, label="1.0"):
    '''
    Returns discriminative performance of the prediction results in a dictionary
    '''
    results = defaultdict(lambda: {})
    report = classification_report(y_true, y_pred, output_dict=True)[label]
    for metric in ['precision','recall','f1-score','support']:
        results[metric] = report[metric]
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    results['auc'] = roc_auc_score(y_true, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()   
    results['tn'], results['fp'], results['fn'], results['tp'] = (tn, fp, fn, tp)
    results['dor'] = dor = (tp/fp)/(fn/tn)

    return results




    


