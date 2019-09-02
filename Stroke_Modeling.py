import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3
import pathlib
import numpy as np
from sklearn.metrics import r2_score, brier_score_loss, mean_absolute_error
from collections import defaultdict
import pickle as pickle


target = "STROKE"

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_lvef.csv')
data = Impute().execute(data, imputation_method=config.DEFAULT)

def train_models(data):

    # model building
    train, test = Split().execute(data, test_size=0.3)

    models = Train().execute(train, target=target, algorithms=[config.RANDOM_FOREST,
                                                               config.DECISION_TREE,
                                                               config.GRADIENT_BOOSTING_DECISION_TREE])
    param_grid_lr = {
        "penalty": ['l2'],
        "C": np.logspace(0, 4, 10),
        "solver": ['lbfgs'],
        "max_iter":[1000]
    }

    hyperparams_lr = {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs'
    }

    hyperparams_rf = {
        'n_estimators': 100,
        'max_depth': 32,
        'max_features': 12,
        'min_samples_leaf': 4,
        'min_samples_split': 8,
    }

    hyperparams_dt = {
        'max_depth': 7,
        'class_weight': {0:1, 1:10}
    }

    hyperparams_gb = {
        'learning_rate': 0.1,
        'n_estimators': 49,
        'max_depth': 3
    }

    hyperparams_mp = {
        'activation': 'tanh',
        'solver': 'sgd',
        'alpha': 1e-5,
        'hidden_layer_sizes': (21, 2),
        'max_iter': 500
    }

    models = {}

    #models.update(Train().execute(train, target=target, optimize='no', hyperparams=hyperparams_lr,
     #                             algorithms=[config.LOGISTIC_REGRESSION]))
    models.update(Train().execute(train, target=target, hyperparams=hyperparams_rf,
                                  algorithms=[config.RANDOM_FOREST]))
    models.update(Train().execute(train, target=target, hyperparams=hyperparams_dt,
                                  algorithms=[config.DECISION_TREE]))
    models.update(Train().execute(train, target=target, hyperparams=hyperparams_gb,
                                  algorithms=[config.GRADIENT_BOOSTING_DECISION_TREE]))
    #models.update(Train().execute(train, target=target, optimize='no', hyperparams=hyperparams_mp,
     #                             algorithms=[config.MULTILAYER_PERCEPTRON]))

# ----------------------------------------------------------------------------------------------------------------------

    # model evaluation on original dataset
    results_org = Evaluate().execute(test, target=target, models=models)

    auc_org = defaultdict(lambda: {})
    auc_org_list = []
    cal_org = defaultdict(lambda: {})
    a_cal_org = defaultdict(lambda: {})
    a_cal_org_list = []
    b_cal_org = defaultdict(lambda: {})
    b_cal_org_list = []

    for alg in results_org:
        #auc_org = int(get_discrimination_metrics(**results_org[alg])['auc']*100)
        #auc_org = get_discrimination_metrics(**results_org[alg])['auc'])
        auc_org[alg] = get_discrimination_metrics(**results_org[alg])
        auc_org[alg] = auc_org[alg]['auc']
        auc_org_list.append(auc_org[alg])
        cal_org[alg] = get_calibration_metrics(results_org[alg]["y_true"], results_org[alg]["y_probs"])
        a_cal_org[alg] = cal_org[alg]['intercept']
        a_cal_org_list.append(a_cal_org[alg])
        b_cal_org[alg] = cal_org[alg]['slope']
        b_cal_org_list.append(a_cal_org[alg])

    auc_org = dict(auc_org)  # change default dict to normal dict
    a_cal_org = dict(a_cal_org)
    b_cal_org = dict(b_cal_org)

    print('Test auc:', auc_org_list)
    print('Test a_cal:', a_cal_org_list)

    pickle.dump(train, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\train.pkl', "wb"))
    pickle.dump(test, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\test.pkl', "wb"))
    pickle.dump(models, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\models.pkl', "wb"))
    pickle.dump(results_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\results_org.pkl', "wb"))
    pickle.dump(auc_org_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_org_list.pkl', "wb"))
    pickle.dump(auc_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_org.pkl', "wb"))
    pickle.dump(a_cal_org_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\a_cal_org_list.pkl', "wb"))
    pickle.dump(a_cal_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\a_cal_org.pkl', "wb"))
    pickle.dump(b_cal_org_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\b_cal_org_list.pkl', "wb"))
    pickle.dump(b_cal_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\b_cal_org.pkl', "wb"))

# ----------------------------------------------------------------------------------------------------------------------

    # model evaluation on artificial datasets
    # loading of datasets to compare
    path = pathlib.Path(r'stroke_preprocessed_imputed_lvef_fake') # change path according to yours
    dis = defaultdict(lambda: {})
    cal = defaultdict(lambda: {})
    auc_list = []
    a_cal_list = []
    b_cal_list = []
    auc = {k: [] for k in results_org}  # init dict with lists
    a_cal = {k: [] for k in results_org}
    b_cal = {k: [] for k in results_org}

    # execute evaluations for every dataset
    for entry in path.iterdir():
        test = Load().execute(source=config.FILE, filename=entry)
        results = Evaluate().execute(test, target=target, models=models)

    # get AUC results for each algorithmn
    for alg in results:
        dis[alg] = get_discrimination_metrics(**results[alg])
        cal[alg] = get_calibration_metrics(results_org[alg]["y_true"], results_org[alg]["y_probs"])
        # a list of all AUC values
        auc_list.append(dis[alg]['auc'])
        a_cal_list.append(cal[alg]['intercept'])
        b_cal_list.append(cal[alg]['slope'])
        # dict categorizing AUC for each algorithmn
        auc[alg].append(dis[alg]['auc'])
        a_cal[alg].append(cal[alg]['intercept'])
        b_cal[alg].append(cal[alg]['slope'])

    pickle.dump(results, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\results.pkl', "wb"))
    pickle.dump(auc_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_list.pkl', "wb"))
    pickle.dump(auc, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc.pkl', "wb"))
    pickle.dump(a_cal_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\a_cal_list.pkl', "wb"))
    pickle.dump(a_cal, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\a_cal.pkl', "wb"))
    pickle.dump(b_cal_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\b_cal_list.pkl', "wb"))
    pickle.dump(b_cal, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\b_cal.pkl', "wb"))

    return train, test, models, results_org, auc_org_list, auc_org, a_cal_org_list, a_cal_org, b_cal_org_list, \
           b_cal_org, results, auc_list, auc, a_cal_list, a_cal, b_cal_list, b_cal

#train_models(data)
