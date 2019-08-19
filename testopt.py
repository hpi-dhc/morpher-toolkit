import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3
import pickle as pickle
from collections import defaultdict

import os
import uuid
import pandas as pd
import numpy as np
import json
import pathlib
import inspect

target = "STROKE"

data = Impute().execute(Load().execute(source=config.FILE, filename="stroke_preprocessed_imputed_lvef.csv"), imputation_method=config.DEFAULT)
#test = Impute().execute(Load().execute(source=config.FILE, filename="test"), imputation_method=config.DEFAULT)

train, test = Split().execute(data, test_size=0.3)

param_grid_lr = {
    "penalty": ['none', 'l2'],
    "C": np.logspace(0, 4, 10),
    "solver": ['lbfgs'],
    "max_iter":[10000]
}

hyperparams_rf = {
    'n_estimators': 300,
    'max_depth': 2
}

models = {}
    
models.update(Train().execute(train, target=target, optimize='yes', param_grid=param_grid_lr, algorithms=[config.LOGISTIC_REGRESSION]))
models.update(Train().execute(train, target=target, hyperparams=hyperparams_rf, algorithms=[config.RANDOM_FOREST]))

results = Evaluate().execute(test, target=target, models=models)

for algorithm in results:
    print("Metrics for {}".format(algorithm))
    print(get_discrimination_metrics(**results[algorithm]))
