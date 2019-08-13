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

target = "AKI"

path = pathlib.Path(r'C:\Users\Harry.FreitasDaCruz\Downloads\aki_fake') # change path according to yours

train = Impute().execute(Load().execute(source=config.FILE, filename="train"), imputation_method=config.DEFAULT)
test = Impute().execute(Load().execute(source=config.FILE, filename="test"), imputation_method=config.DEFAULT)

#models = Train().execute(train, target=target, algorithms=[config.LOGISTIC_REGRESSION, config.DECISION_TREE, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])
#pickle.dump(models, open("models", "wb"))

models = pickle.load(open(r'models', "rb"))

#plt.rc('axes', prop_cycle=prop_cycle)

results = Evaluate().execute(test, target=target, models=models)

#fig, axs = plt.subplots(2, 2, figsize=(15,15))
#fig.suptitle('Vertically stacked subplots')
#plot_roc(results, title="Receiver Operating Curve (A)", ax=axs[0,0], legend_loc="lower right")
#plot_prc(results, title="Precision-Recall Curve (B)", ax=axs[0,1], legend_loc="upper right")
#plot_cc({k:v for k,v in models.items() if k == config.RANDOM_FOREST}, train, test, target, title="Calibration Plot (C)", ax=axs[1,0], legend_loc="lower right")
#plot_dc(results, title="Decision Curve (D)", ax=axs[1,1], legend_loc="lower left")
#fig.tight_layout()
#plot_roc(results)

derivation_metrics = defaultdict(lambda: {})
validation_metrics = defaultdict(lambda: defaultdict(lambda: {}))
cohort_ids = defaultdict(lambda: [])
fqns = []

for model in results:
	derivation_metrics[model] = get_discrimination_metrics(**results[model])
	fqns.append(model)

for filename in path.iterdir():
	cohort_id = str(uuid.uuid4())
	test = Impute().execute(Load().execute(source=config.FILE, filename=filename), imputation_method=config.DEFAULT)
	results = Evaluate().execute(test, target=target, models=models)
	cohort_ids[model] = []
	for model in results:
		validation_metrics[model][cohort_id] = get_discrimination_metrics(**results[model])
		cohort_ids[model].append(cohort_id)

# #let's plot it accordingly
# #fig, axs = plt.subplots(1, len(fqns), sharey=True, figsize=(8,8))
# #fig, axs = plt.subplots(1, len(fqns), sharey=True, figsize=(8,4))
fig, axs = plt.subplots(1, len(fqns),  sharey=True,figsize=(4,4))

#let's plot it accordingly
for index in range(0, len(fqns)):
    auc_derivation = derivation_metrics[fqns[index]]["auc"]        
    aucs_validation = [validation_metrics[fqns[index]][cohort_id]["auc"] for cohort_id in validation_metrics[fqns[index]]]        
    axs[index].plot([auc_derivation] * len(cohort_ids[fqns[index]]), '--', color="orange", label='Derivation AUC'.format(derivation_metrics[fqns[index]]["auc"]))        
    axs[index].plot(aucs_validation, 'o', color="gray", markerfacecolor='none')
    axs[index].plot(aucs_validation, 'royalblue', label='Validation AUC')
    axs[index].set_xlabel('Datasets')
    #print label
    #for i, auc in enumerate(aucs_validation):
    #    axs[index].annotate("{0:.2g}".format(auc), (i, auc ), color="gray")

    if index == 0:
    	axs[index].set_ylabel('Area under the Curve')
    
    axs[index].set_title(fqns[index])
    axs[index].grid()
    #axs[index].legend()

plt.legend()

#let's plot it accordingly
# #fig, axs = plt.subplots(1, len(fqns), sharey=True, figsize=(8,8))
# #fig, axs = plt.subplots(1, len(fqns), sharey=True, figsize=(8,4))

#plt.rc('axes', prop_cycle=prop_cycle)

#fig, ax = plt.figure(figsize=(8,8))

#let's plot it accordingly
#fig.tight_layout()
#plot the Decision Curve for the model selected
#auroc_json = mpld3.fig_to_dict(fig=plt.gcf())

plt.show()

#mpld3.fig_to_dict(fig=plt.gcf())
#mpld3.show()

# #for clf_name in models:
# #	results[clf_name]["clf"] = models[clf_name].clf

# #plot_cc(models, train, test, target)
# mpld3.show()

# #print (mpld3.fig_to_dict(fig=plt.gcf()))

# #chart.serve()

#for alg in results:
	#print("Printing metrics for %s" % alg)
	#print (get_clinical_usefulness_metrics(get_discrimination_metrics(**results[alg]), tr=0.03))














