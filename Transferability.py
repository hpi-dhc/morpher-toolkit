import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3
import pathlib

target = "STROKE"

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_0.csv')

data = Impute().execute(data, imputation_method=config.DEFAULT)

train, test = Split().execute(data, test_size=0.3)

models = Train().execute(train, target=target, algorithms=[
	config.LOGISTIC_REGRESSION])#, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])

results = Evaluate().execute(test, target=target, models=models)

for alg in results:
	auc_org = int(get_discrimination_metrics(**results[alg])['auc']*100)


path = pathlib.Path('./Stroke_data_test')
auc = []

for entry in path.iterdir():

	test = Load().execute(source=config.FILE, filename=entry)

	test = Impute().execute(test, imputation_method=config.DEFAULT)

	results = Evaluate().execute(test, target=target, models=models)

	for alg in results:

		test = get_discrimination_metrics(**results[alg])

		auc.append(int(test['auc']*100))

		print('Original AUC:', auc_org)
		print('AUCs:', auc)

dis_to_org = 0
dis_to_mean = 0
r2 = 0

mean_auc = (sum(auc)+auc_org)/(len(auc)+1)

print('Mean AUC:', mean_auc)

for i in auc:
	dis_to_org += (i - auc_org)**2
	dis_to_mean += (i - mean_auc)**2
r2 = 1 - (dis_to_org / dis_to_mean)

print('Distance to original dataset:', dis_to_org)
print('Distance to AUC Mean:', dis_to_mean)
print('R2:', r2)



#print("Printing metrics for %s" % alg)
	#print(get_clinical_usefulness_metrics(get_discrimination_metrics(**results[alg]), tr=0.03))
	#print(get_calibration_metrics(**results[alg]))


