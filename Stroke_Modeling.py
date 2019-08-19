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

train, test = Split().execute(data, test_size=0.3)

param_grid_dt = {
                    "max_depth": range(4,7),
                    "min_samples_split": range(3,7),
                    "min_samples_leaf": range(1, 16)
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
    "max_depth": range(2,10),
    "n_estimators": range(100, 500, 25)
}

param_grid_lr = {
    "penalty": ['l2'],
    "C": np.logspace(0, 4, 10),
    "solver": ['lbfgs']
}

models = {}

models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_dt,
							  algorithms=[config.DECISION_TREE]))
models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_rf,
							  algorithms=[config.RANDOM_FOREST]))
models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_mp,
							  algorithms=[config.MULTILAYER_PERCEPTRON]))
models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_gb,
							  algorithms=[config.GRADIENT_BOOSTING_DECISION_TREE]))
models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_lr,
							  algorithms=[config.LOGISTIC_REGRESSION]))

results= Evaluate().execute(test, target=target, models=models)

for algorithm in results:
    print("Metrics for {}".format(algorithm))
    print(get_discrimination_metrics(**results[algorithm]))