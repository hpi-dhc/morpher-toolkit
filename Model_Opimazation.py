import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3
import numpy as np
import pickle as pickle
from sklearn.ensemble import ExtraTreesClassifier
from collections import Counter
from sklearn.feature_selection import SelectPercentile
import pandas as pd


target = "STROKE"

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_lvef.csv', delimiter=',')

data = Impute().execute(data, imputation_method=config.DEFAULT)

#-----------------------------------------------------------------------------------------------------------------------

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

# Features via Tree

# liste = []
# num_features = 20  # number of wanted features here
#
# for i in range(1000):  # random comparing 1000 times
#     selector2 = ExtraTreesClassifier()
#     selection2 = selector2.fit(X, y)
#     feat_impo = selection2.feature_importances_
#     ind = np.argpartition(-feat_impo, num_features)
#     highest_ind = ind[:num_features].tolist()
#     liste += highest_ind
#
# dic_features = Counter(liste)
#
# not_selected_features2 = []
#
# for i in range(104):  # build list of possible feature indexes (set number of columns)
#     not_selected_features2.append(i)
#
# list_of_features2 = list(dic_features.keys())[:num_features]
#
# not_selected_features2 = [ele for ele in not_selected_features2 if ele not in list_of_features2]  # subtractes list of features
#
# for i, val in enumerate(not_selected_features2):  # +1 because of STROKE column index at 2
#     if val > 1:
#         not_selected_features2[i] += 1
#
# selected_features2.drop(selected_features2.columns[not_selected_features], axis=1, inplace=True)
#
# print(selected_features2.columns.values)

# top 10 results: 4: 1000, 1: 1000, 2: 1000, 27: 1000, 25: 996, 8: 991, 3: 776, 15: 721, 0: 578, 77: 319

#-----------------------------------------------------------------------------------------------------------------------

# Feature selecetion

data = selected_features
data['STROKE'] = y

#-----------------------------------------------------------------------------------------------------------------------

# Training

train, test = Split().execute(data, test_size=0.3)

param_grid_dt = {
                    "max_depth": range(2, 20),
                    "min_samples_split": range(2, 100),
                    "min_samples_leaf": range(2, 100)
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
        'max_iter': [3000],
        'hidden_layer_sizes': [
            (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,),
            (16,), (17,), (18,), (19,), (20,), (21,), (22,), (23,), (24,), (25,), (26,)
        ]
    }
]

param_grid_gb = {
    "max_depth": range(2, 20),
    "n_estimators": range(1, 100)
}

param_grid_lr = {
    "penalty": ['none', 'l1', 'l2'],
    "C": np.logspace(0, 10, 10),
    "solver": ['newton_sg'] #, 'lbfgs', 'liblinear', 'sag', 'saga']
}

models = {}

#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_dt,
#							  algorithms=[config.DECISION_TREE]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_rf,
#							  algorithms=[config.RANDOM_FOREST]))
models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_mp,
							  algorithms=[config.MULTILAYER_PERCEPTRON]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_gb,
#							  algorithms=[config.GRADIENT_BOOSTING_DECISION_TREE]))
#models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_lr,
#							  algorithms=[config.LOGISTIC_REGRESSION]))

results = Evaluate().execute(test, target=target, models=models)

for algorithm in results:
    print("Metrics for {}".format(algorithm))
    print(get_discrimination_metrics(**results[algorithm]))

