import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3

target = "AKI"

data = Load().execute(source=config.FILE, filename="full.csv")

data = Impute().execute(data, imputation_method=config.DEFAULT)

train, test = Split().execute(data, test_size=0.3)

#models = Train().execute(train, target=target, algorithms=[config.DECISION_TREE, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])
#models = Train(None).execute(train, target=target, algorithms=[config.DECISION_TREE, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_TREE, config.LOGISTIC_REGRESSION, config.MULTILAYER_PERCEPTRON])
#models = Train(None).execute(train, target=target, algorithms=[config.DECISION_TREE, config.RANDOM_FOREST])
#models = Train().execute(train, target=target, algorithms=[config.LOGISTIC_REGRESSION])

models = Train().execute(train, target=target, algorithms=[config.LOGISTIC_REGRESSION, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])

results = Evaluate().execute(test, target=target, models=models)

print(results)

plot_dc(results)

plt.show()

#plot_roc(results)


#print (mpld3.fig_to_dict(fig=plt.gcf()))

# #for clf_name in models:
# #	results[clf_name]["clf"] = models[clf_name].clf

# #plot_cc(models, train, test, target)
# mpld3.show()

# #print (mpld3.fig_to_dict(fig=plt.gcf()))

# #chart.serve()

for alg in results:
	print("Printing metrics for %s" % alg)
	print (get_clinical_usefulness_metrics(get_discrimination_metrics(**results[alg]), tr=0.03))














