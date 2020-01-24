import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import calibration_curve
from collections import defaultdict
from scipy.stats import linregress
import math


def get_discrimination_metrics(y_true, y_pred, y_probs, label="1.0"):
    '''
    Returns discriminative performance of the prediction results in a dictionary
    '''
    results = defaultdict(lambda: {})
    report = classification_report(y_true, y_pred, output_dict=True)[label]
    for metric in ['precision', 'recall', 'f1-score', 'support']:
        results[metric] = float(report[metric])
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    results['auc'] = float(roc_auc_score(y_true, y_probs))
    results['n'] = len(y_true)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    results['tn'], results['fp'], results['fn'], results['tp'] = (int(tn), int(fp), int(fn), int(tp))

    results['dor'] = 0.0

    try:
        results['dor'] = float((tp / fp) / (fn / tn))
    except ZeroDivisionError:
        results['dor'] = 0.0  # undefined

    if results['dor'] == math.inf or results['dor'] == -math.inf or math.isnan(results['dor']):
        results['dor'] = 0.0  # undefined

    return dict(results)


def get_calibration_metrics(y_true, y_pred, y_probs, n_bins=10):

    '''
     Returns calibration metrics of the prediction results in a dictionary:
    '''

    results = defaultdict(lambda: {})
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=n_bins)
    slope, intercept, r_value, p_value, std_err = linregress(fraction_of_positives, mean_predicted_value)
    results['calibration_slope'] = slope
    results['calibration_intercept'] = intercept

    return dict(results)


def get_clinical_usefulness_metrics(discrimination_metrics, tr=0.7):
    '''
    Returns clinical usefulness of the prediction results in a dictionary
    Based on:
    Zhang, Z., Rousson, V., Lee, W.-C., Ferdynus, C., Chen, M., Qian, X., … written on behalf of AME Big-Data Clinical Trial Collaborative Group. (2018). Decision curve analysis: a technical note. Annals of Translational Medicine, 6(15), 308–308. https://doi.org/10.21037/atm.2018.07.02
    '''

    results = discrimination_metrics

    tn, fp, fn, tp, n = list([results.get(metric) for metric in ['tn', 'fp', 'fn', 'tp', 'n']])

    '''
    calculate the benefit of treating vs of not treating
    '''
    net_benefit_treated = (tp / n) - (fp / n) * (tr / (1 - tr))
    net_benefit_untreated = (tn / n) - (fn / n) * ((1 - tr) / tr)

    '''
    pi indicates disease prevalence or event rate
    '''
    pi = (tp + fn) / n

    '''
    net benefit for treating all patients in the given threshould, according to disease prevalence
    net_benefit_treated_all = π – (1–π )pt/ (1-pt )
    '''
    net_benefit_treated_all = pi - (1 - pi) * tr / (1 - tr)

    '''
    ADAPT average deviation about the probability threshold
    '''
    adapt = (1 - tr) * net_benefit_treated + tr * net_benefit_untreated

    results = {}

    results['treated'] = net_benefit_treated
    results['treated_all'] = net_benefit_treated_all
    results['untreated'] = net_benefit_untreated
    results['overall'] = net_benefit_treated + net_benefit_untreated
    results['prevalence'] = pi
    results['adapt'] = adapt
    results['n'] = n

    return dict(results)


def get_calibration_metrics(y_true, y_probs, n_bins=10):  # Why the redefinition?

    '''
     Returns calibration metrics of the prediction results in a dictionary:
    '''

    results = defaultdict(lambda: {})
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=n_bins)
    slope, intercept, r_value, p_value, std_err = linregress(fraction_of_positives, mean_predicted_value)
    results['slope'] = slope
    results['intercept'] = intercept

    return dict(results)


def get_weighted_explanations(explanations, friendly_names=None, weighting={'importance': 1, 'support': 1}):

    '''
     Returns a triple of all the important features from a list of explanations
    '''

    # list of available interpretability methods
    methods = list(explanations.keys())

    # list of all features mentioned across all methods
    features = defaultdict(lambda: [])

    weights = {}

    n_weights = {}

    scaler = MinMaxScaler(feature_range=(0.1, 1))

    # first normalize all feature importance values between 0.1 and 1
    for method in methods:

        importances = np.array(list(explanations[method].values())).reshape(-1, 1)
        scaler.fit(importances)
        n_importances = list(scaler.fit_transform(importances).ravel())
        explanations[method].update({key: value for key, value in zip(list(explanations[method].keys()), n_importances)})

        for feat in explanations[method]:
            features[feat].append(explanations[method][feat])

    # calculate mean while retaining the weights
    for feat in features:
        weights[feat] = len(features[feat])
        features[feat] = float(np.mean(features[feat]))

    # normalize weight values also between 0.1 and 1
    scaler.fit(np.array(list(weights.values())).reshape(-1, 1))
    for feat in features:
        n_weights[feat] = np.asscalar(scaler.transform([[weights[feat]]]))
        features[feat] = (features[feat] * weighting['importance'] + n_weights[feat] * weighting['support']) / (weighting['importance'] + weighting['support'])

    names, vals, weights, n_weights = list(features.keys()), list(features.values()), list(weights.values()), list(n_weights.values())
    if friendly_names:
        names = [friendly_names.get(feat_name) or feat_name for feat_name in names]

    exps = list(sorted(zip(names, vals, weights, n_weights), key=lambda x: x[1]))

    return exps
