import altair as alt
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt, mpld3

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error

prop_cycle = (cycler('color', [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2']) + \
                                cycler('linestyle', [(0, ()),(0, (1, 5)),(0, (1, 1)),(0, (5, 5)),(0, (5, 1)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]))

def plot_roc(results):
    '''
    Plots the receiver operating curve of currently loaded results in a new
    window.
    '''
    plt.clf()
    plt.plot((0, 1), (0, 1), 'k--', label=None)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve')
    plt.rc('axes', prop_cycle=prop_cycle)

    for clf_name in results:
        y_true = results[clf_name]["y_true"]
        y_probs = results[clf_name]["y_probs"]
        fpr, tpr, thresh = roc_curve(y_true, y_probs)

        plt.plot(fpr, tpr, label='{0} (AUC={1:.2f})'.format(clf_name, roc_auc_score(y_true, y_probs)))

    plt.legend(fancybox=True, shadow=True)
    
    






