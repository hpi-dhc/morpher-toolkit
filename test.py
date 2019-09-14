import morpher
import morpher.config as config
from morpher.config import algorithms
from morpher.config import imputers
from morpher.config import scalers
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
train = Load().execute(source=config.FILE, filename="train")
train = Impute().execute(train, imputation_method=imputers.DEFAULT)

print("Imputed: ")
print(train.head(10))

train = Scale().execute(train, target, transform_method=scalers.ROBUST)

print("Transformed: ")

print(train.head(10))

test = Impute().execute(Load().execute(source=config.FILE, filename="test"))

#models = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields])
#models, crossval_metrics = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields], crossval=True, n_splits=50)

#results = Evaluate().execute(test, target=target, models=models)

models = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields])

#print(crossval_metrics)

plt.show()





