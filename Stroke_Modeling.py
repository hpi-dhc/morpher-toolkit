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
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
from sklearn.feature_selection import mutual_info_classif, SelectPercentile, chi2
import pandas as pd



target = "STROKE"

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_lvef.csv')

data = Impute().execute(data, imputation_method=config.DEFAULT)

# Extract most important features

selected_features = pd.DataFrame(data)
selected_features2 = pd.DataFrame(data)

#data = data['ELIXHAUSER_SCORE'].abs()

df = data.values

y = df[:, 2]
X = np.delete(df, [2, 5], 1)

# Features from aki

selector = SelectPercentile(chi2, percentile=25)

X_new = selector.fit(X, y)

liste = X_new.get_support()
not_selected_features = []
i = 0

for x in np.nditer(liste):
    if i == 0 or i == 1:
        if not x:
            not_selected_features.append(i)
    elif not x:
        not_selected_features.append(i+1)  # +1 because of STROKE column index at 2
    i += 1

selected_features.drop(selected_features.columns[not_selected_features], axis=1, inplace=True)

print(selected_features.columns.values)

liste = []

for i in range(1000):
    features = ExtraTreesClassifier()
    features.fit(X, y)
    impo = features.feature_importances_
    ind = np.argpartition(-impo, 20)
    highest_ind = ind[:20].tolist()
    liste += highest_ind

dic = Counter(liste)

not_selected_features2 = []
i = 0
for i in range(104):
    not_selected_features2.append(i)

list_of_features2 = list(dic.keys())[:20]

not_selected_features2 = [ele for ele in not_selected_features2 if ele not in list_of_features2]

for i, val in enumerate(not_selected_features2):
    if val > 1:
        not_selected_features2[i] += 1

selected_features2.drop(selected_features2.columns[not_selected_features], axis=1, inplace=True)

print(selected_features2.columns.values)

# top 10 results: 4: 1000, 1: 1000, 2: 1000, 27: 1000, 25: 996, 8: 991, 3: 776, 15: 721, 0: 578, 77: 319

# for ind, column in enumerate(data.columns[::-1]):
#     if ind == 4 or ind == 1 or ind == 2 or ind == 27 or ind == 25 or ind == 8 or ind == 3 or ind == 15 or ind == 0 or ind == 77:
#         pass
#     else:
#         data = data.drop([column])
#data = data.reset_index()
#data = data.drop(['index'], axis=1)
#print(data)

train, test = Split().execute(data, test_size=0.3)

param_grid_dt = {
                    "max_depth": range(2, 20),
                    "min_samples_split": range(2, 20),
                    "min_samples_leaf": range(1, 20)
                    #"min_impurity_decrease": np.arange(0.0, 0.3, 0.025)
                }

param_grid_rf = {

    'bootstrap': [True],
    'max_depth': [2, 4, 8, 16],
    'max_features': [2, 4, 8],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300]
}

param_grid_mp = [
    {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,),
            (16,), (17,), (18,), (19,), (20,), (21,)
        ]
    }
]

param_grid_gb = {
    "max_depth": range(2, 10),
    "n_estimators": range(1, 85, 1)
}

param_grid_lr = {
    "penalty": ['none', 'l1', 'l2'],
    "C": np.logspace(0, 10, 10),
    "solver": ['newton_sg'] #, 'lbfgs', 'liblinear', 'sag', 'saga']
}

#models = {}

#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_dt,
#							  algorithms=[config.DECISION_TREE]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_rf,
#							  algorithms=[config.RANDOM_FOREST]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_mp,
#							  algorithms=[config.MULTILAYER_PERCEPTRON]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_gb,
#							  algorithms=[config.GRADIENT_BOOSTING_DECISION_TREE]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_lr,
#							  algorithms=[config.LOGISTIC_REGRESSION]))

#results= Evaluate().execute(test, target=target, models=models)

#for algorithm in results:
#    print("Metrics for {}".format(algorithm))
#    print(get_discrimination_metrics(**results[algorithm]))