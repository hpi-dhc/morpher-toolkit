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
        results[metric] = float(report[metric])
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    results['auc'] = float(roc_auc_score(y_true, y_probs))
    results['n'] = y_true.shape[0]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results['tn'], results['fp'], results['fn'], results['tp'] = (int(tn), int(fp), int(fn), int(tp))

    try:
        results['dor'] = float((tp/fp)/(fn/tn))
    except ZeroDivisionError as e:
        results['dor'] = 0.0

    return dict(results)

def get_clinical_usefulness_metrics(discrimination_metrics, tr=0.7):
    '''
    Returns clinical usefulness of the prediction results in a dictionary
    '''    

    results = discrimination_metrics

    tn, fp, fn, tp, n = list([results.get(metric) for metric in ['tn', 'fp', 'fn', 'tp', 'n']])

    '''
    calculate the benefit of treating vs of not treating
    '''
    net_benefit_treated = (tp/n) - (fp/n) * (tr/(1-tr))
    net_benefit_untreated = (tn/n) - (fn/n) * ((1-tr)/tr)

    '''
    pi indicates disease prevalence or event rate
    '''
    pi = (tp + fn) / n

    '''
    net benefit for treating all patients in the given threshould, according to disease prevalence
    π – (1–π )pt/ (1-pt )
    '''
    net_benefit_treated_all = pi - (1-pi) * tr / (1-tr)


    '''
    ADAPT average deviation about the probability threshold
    '''
    adapt = (1 - tr) * net_benefit_treated + tr * net_benefit_untreated

    results['treated'] = net_benefit_treated
    results['treated_all'] = net_benefit_treated_all
    results['untreated'] = net_benefit_untreated
    results['overall'] = net_benefit_treated + net_benefit_untreated
    results['prevalence'] = pi
    results['adapt'] = pi

    return results






    


