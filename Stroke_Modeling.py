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

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_lvef.csv', delimiter=',')
data = Impute().execute(data, imputation_method=config.DEFAULT)

# Extract most important features

selected_features = pd.DataFrame(data)
selected_features2 = pd.DataFrame(data)

data['ELIXHAUSER_SCORE'] = data['ELIXHAUSER_SCORE'].abs()

data = data[data['AGE_AT_ADMISSION'] < 99]

y = data['STROKE']
X = data.drop(['STROKE', 'OTHER_NEUROLOGICAL', 'PARALYSIS'], axis=1)

# Features via Mutual Information

selector = SelectPercentile(mutual_info_classif, percentile=25)

selection = selector.fit(X, y)

sup = selection.get_support()

selected_features = data[X.columns[sup]]

print(selected_features.columns)

# Feature selecetion

data = selected_features
data['STROKE'] = y

train, test = Split().execute(data, test_size=0.3)

param_grid_lr = {
    "penalty": ['l2'],
    "C": np.logspace(0, 4, 10),
    "solver": ['lbfgs'],
    "max_iter": [1000]
}

hyperparams_lr = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'liblinear'
}

hyperparams_rf = {
    'n_estimators': 100,
    'max_depth': 16,
    'max_features': 2,
    'min_samples_leaf': 5,
    'min_samples_split': 16
}

hyperparams_dt = {
    'max_depth': 9,
    "min_samples_split": 2,
    "min_samples_leaf": 51
}

hyperparams_gb = {
    'n_estimators': 54,
    'max_depth': 3
}

hyperparams_mp = {
    'activation': 'relu',
    'solver': 'adam',
    'hidden_layer_sizes': (14, ),
    'max_iter': 3000
}

models = {}

models.update(Train().execute(train, target=target, hyperparams=hyperparams_lr,
                              algorithms=[config.LOGISTIC_REGRESSION]))
models.update(Train().execute(train, target=target, hyperparams=hyperparams_rf,
                              algorithms=[config.RANDOM_FOREST]))
models.update(Train().execute(train, target=target, hyperparams=hyperparams_dt,
                              algorithms=[config.DECISION_TREE]))
models.update(Train().execute(train, target=target, hyperparams=hyperparams_gb,
                              algorithms=[config.GRADIENT_BOOSTING_DECISION_TREE]))
models.update(Train().execute(train, target=target, hyperparams=hyperparams_mp,
                              algorithms=[config.MULTILAYER_PERCEPTRON]))

# ----------------------------------------------------------------------------------------------------------------------



# model evaluation on original dataset
results_org = Evaluate().execute(test, target=target, models=models, print_performance=True)

auc_org = defaultdict(lambda: {})
auc_org_list = []
cal_org = defaultdict(lambda: {})
a_cal_org = defaultdict(lambda: {})
a_cal_org_list = []
b_cal_org = defaultdict(lambda: {})
b_cal_org_list = []
brier_cal_org = defaultdict(lambda: {})
brier_cal_org_list = []
nb_org = defaultdict(lambda: {})
nb_org_list = []

for alg in results_org:
    #auc_org = int(get_discrimination_metrics(**results_org[alg])['auc']*100)
    #auc_org = get_discrimination_metrics(**results_org[alg])['auc'])
    auc_org[alg] = get_discrimination_metrics(**results_org[alg])
    auc_org[alg] = auc_org[alg]['auc']
    auc_org_list.append(auc_org[alg])
    cal_org[alg] = get_calibration_metrics(results_org[alg]["y_true"], results_org[alg]["y_probs"])
    nb_org[alg] = (get_clinical_usefulness_metrics(get_discrimination_metrics(**results_org[alg])))
    a_cal_org[alg] = cal_org[alg]['intercept']
    a_cal_org_list.append(a_cal_org[alg])
    b_cal_org[alg] = cal_org[alg]['slope']
    b_cal_org_list.append(a_cal_org[alg])
    brier_cal_org[alg] = cal_org[alg]['brier_calibration']
    brier_cal_org_list.append(brier_cal_org[alg])
    nb_org[alg] = nb_org[alg]['treated']
    nb_org_list.append(nb_org[alg])


auc_org = dict(auc_org)  # change default dict to normal dict
a_cal_org = dict(a_cal_org)
b_cal_org = dict(b_cal_org)
brier_cal_org = dict(brier_cal_org)
nb_org = dict(nb_org)

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
pickle.dump(brier_cal_org_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\brier_cal_org_list.pkl', "wb"))
pickle.dump(brier_cal_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\brier_cal_org.pkl', "wb"))
pickle.dump(nb_org_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\nb_org_list.pkl', "wb"))
pickle.dump(nb_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\nb_org.pkl', "wb"))

# ----------------------------------------------------------------------------------------------------------------------

# model evaluation on artificial datasets
# loading of datasets to compare
path = pathlib.Path(r'stroke_preprocessed_imputed_lvef_fake') # change path according to yours
results_list = []

# execute evaluations for every dataset
for entry in path.iterdir():
    test = Load().execute(source=config.FILE, filename=entry)
    y_2 = test['STROKE']
    test = test[X.columns[sup]]
    test['STROKE'] = y_2

    results = Evaluate().execute(test, target=target, models=models)
    results_list.append(results)

# initialize variables
dis = defaultdict(lambda: {})
cal = defaultdict(lambda: {})
cu = defaultdict(lambda: {})
auc_list = []
a_cal_list = []
b_cal_list = []
brier_cal_list = []
nb_list = []
auc = {k: [] for k in results}  # init dict with lists
a_cal = {k: [] for k in results}
b_cal = {k: [] for k in results}
brier_cal = {k: [] for k in results}
nb = {k: [] for k in results}

# get results for each algorithmn
for result in results_list:
    for alg in result:
        dis[alg] = get_discrimination_metrics(**result[alg])
        cal[alg] = get_calibration_metrics(result[alg]["y_true"], result[alg]["y_probs"])
        cu[alg] = get_clinical_usefulness_metrics(dis[alg])
        # a list of all metrics values
        auc_list.append(dis[alg]['auc'])
        a_cal_list.append(cal[alg]['intercept'])
        b_cal_list.append(cal[alg]['slope'])
        brier_cal_list.append(cal[alg]['brier_calibration'])
        nb_list.append(cu[alg]['treated'])
        # dict categorizing for each algorithmn
        auc[alg].append(dis[alg]['auc'])
        a_cal[alg].append(cal[alg]['intercept'])
        b_cal[alg].append(cal[alg]['slope'])
        brier_cal[alg].append(cal[alg]['brier_calibration'])
        nb[alg].append(cu[alg]['treated'])

pickle.dump(results_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\results.pkl', "wb"))
pickle.dump(auc_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_list.pkl', "wb"))
pickle.dump(auc, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc.pkl', "wb"))
pickle.dump(a_cal_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\a_cal_list.pkl', "wb"))
pickle.dump(a_cal, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\a_cal.pkl', "wb"))
pickle.dump(b_cal_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\b_cal_list.pkl', "wb"))
pickle.dump(b_cal, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\b_cal.pkl', "wb"))
pickle.dump(brier_cal_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\brier_cal_list.pkl', "wb"))
pickle.dump(brier_cal, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\brier_cal.pkl', "wb"))
pickle.dump(nb_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\nb_list.pkl', "wb"))
pickle.dump(nb, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\nb.pkl', "wb"))


for alg in results_org:
    print(cal[alg])

plt.figure(1)
plot_cc(models, train, test, target='STROKE')
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.show()

