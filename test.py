import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3
import pickle as pickle

target = "AKI"

data = Load().execute(source=config.FILE, filename="full.csv")

data = Impute().execute(data, imputation_method=config.DEFAULT)

train, test = Split().execute(data, test_size=0.3)

#models = Train().execute(train, target=target, algorithms=[config.DECISION_TREE, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])
#models = Train(None).execute(train, target=target, algorithms=[config.DECISION_TREE, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_TREE, config.LOGISTIC_REGRESSION, config.MULTILAYER_PERCEPTRON])
#models = Train(None).execute(train, target=target, algorithms=[config.DECISION_TREE, config.RANDOM_FOREST])
#models = Train().execute(train, target=target, algorithms=[config.LOGISTIC_REGRESSION])

#models = Train().execute(train, target=target, algorithms=[config.LOGISTIC_REGRESSION, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])

#pickle.dump(models, open("models", "wb"))

plt.rc('axes', prop_cycle=prop_cycle)


models = pickle.load(open("models", "rb"))

results = Evaluate().execute(test, target=target, models=models)

#fig, axs = plt.subplots(2, 2, figsize=(15,15))

#fig.suptitle('Vertically stacked subplots')

#plot_roc(results, title="Receiver Operating Curve (A)", ax=axs[0,0])
#plot_prc(results, title="Precision-Recall Curve (B)", ax=axs[0,1])
#plot_cc({k:v for k,v in models.items() if k == config.RANDOM_FOREST}, train, test, target, title="Calibration Plot (C)", ax=axs[1,0])
#plot_dc(results, title="Decision Curve (D)", ax=axs[1,1])

#fig.tight_layout()
#plt.show()

plot_roc(results)

#mpld3.fig_to_dict(fig=plt.gcf())
mpld3.show()

# #for clf_name in models:
# #	results[clf_name]["clf"] = models[clf_name].clf

# #plot_cc(models, train, test, target)
# mpld3.show()

# #print (mpld3.fig_to_dict(fig=plt.gcf()))

# #chart.serve()

#for alg in results:
	#print("Printing metrics for %s" % alg)
	#print (get_clinical_usefulness_metrics(get_discrimination_metrics(**results[alg]), tr=0.03))














