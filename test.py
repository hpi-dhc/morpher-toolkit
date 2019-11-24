import morpher
import morpher.config as config
from morpher.config import algorithms
from morpher.config import imputers
from morpher.config import scalers
from morpher.config import explainers
from morpher.config import encoders
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

data = pd.DataFrame({'pet':['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],'children': [4., 6, 3, 3, 2, 3, 5, 4],'salary':   [90., 24, 44, 27, 32, 59, 36, 27]})

print(data)

data = Transform().execute(data, transforms=[('pet',encoders.BINARIZER), (['salary'], scalers.DEFAULT)])

print(data)
exit()

#train = Transform().execute(train, transforms={'GENDER' : encoders.BINARIZER})

train = Scale().execute(train, target, transform_method=scalers.ROBUST)

print("Transformed: ")

print(train.head(10))

test = Impute().execute(Load().execute(source=config.FILE, filename="test"))

#models = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields])
#models, crossval_metrics = Train().execute(train, target=target, algorithms=[getattr(algorithms, algorithm) for algorithm in algorithms._fields], crossval=True, n_splits=50)


models = Train().execute(train, target=target, algorithms=[algorithms.GBDT])

#explanations = Explain().execute(train, models=models, target=target, explainers=[explainers.MIMIC, explainers.FEAT_CONTRIB, explainers.SHAP], exp_kwargs={'test':test})
#explanations = Explain().execute(train, models=models, target=target, explainers=[explainers.SHAP, explainers.MIMIC, explainers.FEAT_CONTRIB, explainers.LIME], exp_kwargs={'test':test})
#explanations = Explain().execute(train, models=models, target=target, explainers=[explainers.SHAP, explainers.MIMIC, explainers.FEAT_CONTRIB], exp_kwargs={'test':test})
explanations = Explain().execute(train, models=models, target=target, explainers=[explainers.SHAP, explainers.MIMIC, explainers.FEAT_CONTRIB], exp_kwargs={'test':test, 'print_exps':True})
#pickle.dump(explanations, open("explanations.pkl", "wb"))

#results = Evaluate().execute(test, target=target, models=models)
#print(explanations[algorithms.GBDT][explainers.LIME][0])
# for model in models:
# 	for explainer in explanations[model]:		
# 		for exp in explanations[model][explainer]:			
# 			for key,value in exp.items():
# 				print(f"{key}:{value}")
# 		print("\n")
#print(crossval_metrics)
#plot_feat_importances(explanations[algorithms.GBDT][explainers.SHAP][0], friendly_names={"ALCOHOL_ABUSE" : "AA Member?"})

print(get_weighted_explanations(explanations[algorithms.GBDT]))

#plt.tight_layout()
#plt.show()
#mpld3.show()





