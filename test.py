import morpher
import morpher.config as config
from morpher.config import algorithms
from morpher.config import imputers
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

target = "AKI"

train = Impute().execute(Load().execute(source=config.FILE, filename="train"), imputation_method=imputers.DEFAULT)
test = Impute().execute(Load().execute(source=config.FILE, filename="test"), imputation_method=imputers.DEFAULT)
#models = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields])
models, crossval_metrics = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields], crossval=True, n_splits=50)

results = Evaluate().execute(test, target=target, models=models)

print(crossval_metrics)

plt.show()





