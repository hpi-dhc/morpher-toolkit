import pandas as pd
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt, mpld3

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error, precision_recall_curve, average_precision_score, auc

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

prop_cycle = (cycler('color', [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2']) + \
                                cycler('linestyle', [(0, ()),(0, (1, 5)),(0, (1, 1)),(0, (5, 5)),(0, (5, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]))

from morpher.algorithms import *

from morpher.metrics import *

def plot_roc(results, title="Receiver Operating Curve", ax=None, figsize=None):
    '''
    Plots the receiver operating curve of currently loaded results in a new
    window.
    '''
    if not ax:
        ax = plt.gca()
    
    if figsize:
        plt.clf()
        fig = plt.figure(figsize=figsize)

    ax.plot((0, 1), (0, 1), 'k--', label=None)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    plt.rc('axes', prop_cycle=prop_cycle)

    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_probs = results[clf_name]["y_probs"]
        fpr, tpr, thresh = roc_curve(y_true, y_probs)

        ax.plot(fpr, tpr, label='{0} (AUC={1:.2f})'.format(clf_name, roc_auc_score(y_true, y_probs)))

    ax.legend(loc="lower right", fancybox=True, shadow=True)

def plot_prc(results, title="Precision-Recall Curve", ax=None, figsize=None):
    '''
    Plots the precision recall curve currently loaded results in a new
    window.
    '''
    if not ax:
        ax = plt.gca()
    
    if figsize:
        plt.clf()
        fig = plt.figure(figsize=figsize)

    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    plt.rc('axes', prop_cycle=prop_cycle)

    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_probs = results[clf_name]["y_probs"]
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ax.step(recall, precision, label='{0} (AP={1:.2f})'.format(clf_name, average_precision_score(y_true, y_probs)), where='post')        

    ax.plot([0, 1], [0.5, 0.5], label='No skill', color='lightgray', linestyle='--')

    ax.legend(loc="upper right", fancybox=True, shadow=True)
    ax.autoscale(enable=True) 


def plot_cc(models, train_data, test_data, target, title="Calibration Plot", ax=None, figsize=None):
    '''
    Plots calibration curve, we need the original train dataset to do this (!)
    '''
    if not models:
        raise AttributeError("No models available")

    if not ax:
        ax = plt.gca()
    
    if figsize:
        plt.clf()
        fig = plt.figure(figsize=figsize)

    y_train = train_data[target]
    X_train = train_data.drop(target, axis=1)

    y_test = test_data[target]
    X_test = test_data.drop(target, axis=1)

    plt.rc('axes', prop_cycle=prop_cycle)

    for clf_name in models:

        clf = models[clf_name].clf
        
        calibrated_clf_sig = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
        calibrated_clf_sig.fit(X_train, y_train)
        
        calibrated_clf = CalibratedClassifierCV(clf, cv='prefit', method='isotonic')
        calibrated_clf.fit(X_train, y_train)
        
        for clf, name in [(clf, clf_name),(calibrated_clf, clf_name + ' + isotonic'), (calibrated_clf_sig, clf_name + ' + sigmoid')]:
            
            y_probs = clf.predict_proba(X_test)
            score = brier_score_loss(y_test, y_probs[:,1], pos_label=y_test.max())
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_probs[:,1], n_bins=10)
            ax.plot(mean_predicted_value, fraction_of_positives, label="%s (Brier=%1.3f)" % (name, score), marker="s")

            score = brier_score_loss(y_test, y_probs[:,1], pos_label=y_test.max())  
            print("*** Brier for %s: %1.3f" % (name, score))


    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right", fancybox=True, shadow=True)
    ax.set_title(title)

    print("*** Model calibration performed.\n")

def plot_dc(results, tr_start=0.01, tr_end=0.99, tr_step=0.01, metric_type="treated", ymin=-0.1, title="Decision Curve", ax=None, figsize=None):
    '''
    Plots decision curve for treating a patient based on the predicted probability
    
    Params:
    
    results               outputs of a given classifier
    tr_start              start of thresholds
    tr_end                end of thresholds
    tr_end                step_size of thresholds
    metric_type           type of metric to be plotted, treated, untreated or adapt
    ymin                  mininum net benefit to be plotted
    
    '''
    if not results:
        raise AttributeError("No results available")

    if not ax:
        ax = plt.gca()
    
    if figsize:
        plt.clf()
        fig = plt.figure(figsize=figsize)

    ax.set_xlabel('Threshold Probability')
    ax.set_label('Net Benefit')
    ax.set_title(title)
    plt.rc('axes', prop_cycle=prop_cycle)
    ax.set_xticks(np.arange(0, 1.25, step=0.25))
    
    tr_probs = np.arange(tr_start, tr_end + tr_step, tr_step)
    
    '''
    Necessary to decide where to fix ymax
    '''
    ymax = 0.0
    current_ymax = 0.0
    
    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_pred = results[clf_name]["y_pred"]
        y_probs = results[clf_name]["y_probs"]

        discrimination_metrics = get_discrimination_metrics(y_true, y_pred, y_probs)
        net_benefit = [get_clinical_usefulness_metrics(discrimination_metrics, tr)[metric_type] for tr in tr_probs]
        net_benefit_treated_all = [get_clinical_usefulness_metrics(discrimination_metrics, tr)["treated_all"] for tr in tr_probs]
        
        ax.plot(tr_probs, net_benefit, label='{0} (ANBC={1:.2f})'.format(clf_name, auc(tr_probs, net_benefit)))   

        '''
        Necessary to decide where to fix ymax
        '''
        current_max = np.amax(net_benefit + net_benefit_treated_all)
        if current_max > ymax:
            ymax = current_max
    
    ax.plot(tr_probs, net_benefit_treated_all, label=metric_type + ' (all)')    
    ax.axhline(y=0.0, color='gray', linestyle='--', label=metric_type + ' (none)')
    ax.legend(loc="upper right", fancybox=True, shadow=True)

    '''
    Define plotting limits
    '''
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([0,1])


    
    






