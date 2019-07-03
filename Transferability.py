import morpher
import morpher.config as config
from morpher.jobs import *
from morpher.metrics import *
from morpher.plots import *
import matplotlib.pyplot as plt, mpld3
import pathlib
import numpy as np
from sklearn.metrics import r2_score, brier_score_loss, mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

target = "STROKE"

data = Load().execute(source=config.FILE, filename='stroke_preprocessed.csv')

data = Impute().execute(data, imputation_method=config.DEFAULT)

train, test = Split().execute(data, test_size=0.3)

models = Train().execute(train, target=target, algorithms=[
	config.LOGISTIC_REGRESSION])#, config.RANDOM_FOREST, config.GRADIENT_BOOSTING_DECISION_TREE, config.MULTILAYER_PERCEPTRON])

results = Evaluate().execute(test, target=target, models=models)

for alg in results:
	#auc_org = int(get_discrimination_metrics(**results[alg])['auc']*100)
	auc_org = get_discrimination_metrics(**results[alg])['auc']


path = pathlib.Path('./stroke_preprocessed_imputed')
auc = []

for entry in path.iterdir():

	test = Load().execute(source=config.FILE, filename=entry)
	results = Evaluate().execute(test, target=target, models=models)

	for alg in results:

		test = get_discrimination_metrics(**results[alg])

		#auc.append(int(test['auc']*100))
		auc.append(test['auc'])

print('Original AUC:', auc_org)
print('AUCs:', auc)

dis_to_org = 0
dis_to_mean = 0
r2 = 0

#mean_auc = (sum(auc)+auc_org)/(len(auc)+1)
mean_auc = (sum(auc))/(len(auc))

print('Mean AUC:', mean_auc)

for auc_i in auc:
	dis_to_org += (auc_i - auc_org)**2
	dis_to_mean += (auc_org - mean_auc)**2

r2 = 1 - (dis_to_org / dis_to_mean)

print('Distance to original dataset:', dis_to_org)
print('Distance to AUC Mean:', dis_to_mean)
print('R2:', r2)
print('R2 (sklearn):', r2_score([auc_org] * len(auc), auc))
print('Brier:', brier_score_loss([auc_org] * len(auc), auc))
print('Mean absolute error:', mean_absolute_error([auc_org] * len(auc), auc))
print('Mean absolute percentage error:', mean_absolute_percentage_error([auc_org] * len(auc), auc))

plt.xlabel('Datasets')
plt.ylabel('Area Under the ROC Curve')
plt.title('Receiver Operating Curve')

plt.plot(auc, 'ro')
plt.plot(auc)
plt.plot([auc_org] * len(auc))
plt.show()


#print("Printing metrics for %s" % alg)
	#print(get_clinical_usefulness_metrics(get_discrimination_metrics(**results[alg]), tr=0.03))
	#print(get_calibration_metrics(**results[alg]))


