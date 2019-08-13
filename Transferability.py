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


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

target = "STROKE"

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_lvef.csv')

data = Impute().execute(data, imputation_method=config.DEFAULT)

train, test = Split().execute(data, test_size=0.3)

models = Train().execute(train, target=target, algorithms=[config.LOGISTIC_REGRESSION, config.RANDOM_FOREST,
														   config.DECISION_TREE, config.GRADIENT_BOOSTING_DECISION_TREE,
														   config.MULTILAYER_PERCEPTRON])

results_org = Evaluate().execute(test, target=target, models=models)

auc_org = defaultdict(lambda: {})
auc_org_list = []

for alg in results_org:
	#auc_org = int(get_discrimination_metrics(**results_org[alg])['auc']*100)
	#auc_org = get_discrimination_metrics(**results_org[alg])['auc'])
	auc_org[alg] = get_discrimination_metrics(**results_org[alg])
	auc_org[alg] = auc_org[alg]['auc']
	auc_org_list.append(auc_org[alg])
auc_org = dict(auc_org)  # change default dict to normal dict
#
#
# pickle.dump(train, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\train.pkl', "wb"))
# pickle.dump(test, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\test.pkl', "wb"))
# pickle.dump(models, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\models.pkl', "wb"))
# pickle.dump(results_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\results_org.pkl', "wb"))
# pickle.dump(auc_org_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_org_list.pkl', "wb"))
# pickle.dump(auc_org, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_org.pkl', "wb"))

# train = pickle.load(open(r'results_performance\train.pkl', "rb"))
# test = pickle.load(open(r'results_performance\test.pkl', "rb"))
# models = pickle.load(open(r'results_performance\models.pkl', "rb"))
# results_org = pickle.load(open(r'results_performance\results_org.pkl', "rb"))
# auc_org_list = pickle.load(open(r'results_performance\auc_list.pkl', "rb"))
# auc_org = pickle.load(open(r'results_performance\auc_org.pkl', "rb"))

# loading of datasets to compare
path = pathlib.Path(r'stroke_preprocessed_imputed_lvef_fake') # change path according to yours
dis = defaultdict(lambda: {})
auc_list = []
auc = {k: [] for k in results_org}  # init dict with lists

# execute evaluations for every dataset
for entry in path.iterdir():
	test = Load().execute(source=config.FILE, filename=entry)
	results = Evaluate().execute(test, target=target, models=models)

	# get AUC results for each algorithmn
	for alg in results:
		dis[alg] = get_discrimination_metrics(**results[alg])

		# a list of all AUC values
		auc_list.append(dis[alg]['auc'])
		# dict categorizing AUC for each algorithmn
		auc[alg].append(dis[alg]['auc'])

# pickle.dump(results, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\results.pkl', "wb"))
# pickle.dump(auc_list, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc_list.pkl', "wb"))
# pickle.dump(auc, open(r'C:\Users\Margaux\Documents\GitHub\morpher\results_performance\auc.pkl', "wb"))

# results = pickle.load(open(r'results_performance\results.pkl', "rb"))
# auc_list = pickle.load(open(r'results_performance\auc_list.pkl', "rb"))
# auc = pickle.load(open(r'results_performance\auc.pkl', "rb"))

# mean AUC calculations
mean_auc_total = (sum(auc_list))/(len(auc_list)) # mean without original AUC
#mean_auc = (sum(auc)+auc_org)/(len(auc)+1)
mean_auc_org_total = (sum(auc_org_list))/(len(auc_org_list)) # calculates an average as AUC original score of all algorithmns

mean_auc = defaultdict(lambda: {})
for alg in results:
	mean_auc[alg] = (sum(auc[alg]))/(len(auc[alg]))


# distance to original and mean AUC and R2

# for total AUC
dis_to_org_total = 0
dis_to_mean_total = 0
r2_total = 0

for auc_i in auc_list:
	dis_to_org_total += (auc_i - mean_auc_org_total)**2
	#dis_to_mean_total += (auc_org - mean_auc)**2 ?? Harry why did you use the original auc here??
	dis_to_mean_total += (auc_i - mean_auc_total)**2
r2_total = 1 - (dis_to_org_total / dis_to_mean_total)

# for each algorithmn
dis_mean = 0
dis_org = 0
dis_to_org = defaultdict(lambda: {})
dis_to_mean = defaultdict(lambda: {})
r2 = defaultdict(lambda: {})

for alg in results:
	for auc_i in auc[alg]:
		dis_org += (auc_i - auc_org[alg]) ** 2
		dis_mean += (auc_i - mean_auc[alg]) ** 2
	dis_to_org[alg] = dis_org
	dis_to_mean[alg] = dis_mean
	r2[alg] = 1 - (dis_to_org[alg] / dis_to_mean[alg])

# variance regarding original AUC
var = defaultdict(lambda: {})
var_total = 0
for alg in results:
	var[alg] = sum((i - auc_org[alg]) ** 2 for i in auc[alg]) / len(auc[alg])
var_total = sum((i - mean_auc_org_total) ** 2 for i in auc_list) / len(auc_list)


# Output

# For each algorithmn
for alg in results:
	print('Transferability Performances Measures for', alg, ':')
	print('\n')
	print('Mean AUC for', alg, ':', round(mean_auc[alg], 3))
	print('Distance to original dataset:', round(dis_to_org[alg], 3))
	print('Distance to AUC Mean:', round(dis_to_mean[alg], 3))
	print('R2:', round(r2[alg], 3))
	print('R2 (sklearn):',
		  round(r2_score([auc_org[alg]] * len(auc[alg]), auc[alg]), 2))  # the values are extremly different
	print('Brier:', round(brier_score_loss([auc_org[alg]] * len(auc[alg]), auc[alg]), 3))
	print('Mean absolute error:', round(mean_absolute_error([auc_org[alg]] * len(auc[alg]), auc[alg]), 3))
	print('Mean absolute percentage error:',
		  round(mean_absolute_percentage_error([auc_org[alg]] * len(auc[alg]), auc[alg]), 3), '%', )
	# Percentage difference between the mean of AUCs and the original AUC
	print('Transferability for', alg, ':', 100 - (int(np.abs(auc_org[alg] - mean_auc[alg]) * 100)), "%")
	print('Variance of', alg, 'AUC in regard to orginal AUC:', round(var[alg], 5))
	print('\n')

# Total Scores
print('Total Scores:')
print('Mean AUC:', mean_auc_total)
print('Distance to original dataset:', dis_to_org_total)
print('Distance to AUC Mean:', dis_to_mean_total)
print('R2:', r2_total)
print('R2 (sklearn):', r2_score([mean_auc_org_total] * len(auc_list), auc_list)) # the values are extremly different, often zero
print('Brier:', brier_score_loss([mean_auc_org_total] * len(auc_list), auc_list))
print('Mean absolute error:', mean_absolute_error([mean_auc_org_total] * len(auc_list), auc_list))
print('Mean absolute percentage error:', mean_absolute_percentage_error([mean_auc_org_total] * len(auc_list), auc_list), '%')
# Percentage difference between the mean of AUCs and the original AUC
print('Transferability_overall:', 100 - (int(np.abs(mean_auc_org_total - mean_auc_total) * 100)), "%")
print('Variance of AUC in regard to orginal AUC:', var_total)
print('\n')

# Calibrations and Clinical Usefulness metrics
#for alg in results_org:
	#print("Printing metrics for %s" % alg)
	#print(get_discrimination_metrics(**results_org[alg]))
	# print(get_calibration_metrics(**results_org[alg]))
	# print(get_clinical_usefulness_metrics(get_discrimination_metrics(**results_org[alg]), tr=0.03))
	#print('\n')


# Plot of orginal AUC against all AUCs (maybe we should also use the means here, or its not really comparable)
# plt.figure(1)
# plt.xlabel('Datasets')
# plt.ylabel('Area Under the ROC Curve')
# plt.title('Total Receiver Operating Curve')
#
# plt.plot(auc_list, 'ro')
# plt.plot(auc_list)
# plt.plot([mean_auc_org_total] * len(auc_list))
# plt.show()

# Plotting with categorical variables
n = 101  # subplot number must be a three digit number
n += len(results) * 10  # second digit shows number of graph in one row
subplot_ylabel = n
plt.figure(2, figsize=(18.5, 7))

for alg in results:
	plt.subplot(n)
	plt.plot(auc[alg], 'ro')
	plt.plot(auc[alg], label='AUC\'s')
	plt.plot([auc_org[alg]] * len(auc[alg]), label='AUC original dataset')
	plt.title('Area Under the ROC Curv')


	plt.xlabel('Datasets')
	if n == subplot_ylabel:  # shows only the label on the left
		plt.ylabel(alg)
	plt.title(alg)

	n += 1
#plt.subplots_adjust(left=0.1)
plt.legend(bbox_to_anchor=(0.8, 1.05))
plt.show()

# seperates plots of each algorithmn
# i = 3
#
# for alg in results:
# 	plt.figure(i)
# 	plt.xlabel('Datasets')
# 	plt.ylabel('Area Under the ROC Curve')
# 	plt.title('Receiver Operating Curve for ' + str(alg))
#
# 	plt.plot(auc[alg], 'ro')
# 	plt.plot(auc[alg])
# 	plt.plot([auc_org[alg]] * len(auc[alg]))
# 	plt.show()
#
# 	i += 1

# plt.figure(4)
# plot_roc(results_org)
# plt.show()


# # plot Decision Curve
# plt.figure(3)
# plot_dc(results_org)
# plt.show()
#
# plot Calibration Curve
# plt.figure(4)
# plot_cc(models, train, test, target)
# plt.legend(bbox_to_anchor=(1.3, 1.05))
# plt.show()


# Transferability metrics for presentation
for alg in results:
	print('Transferability Performances Measures for', alg, ':')
	print('')
	print('Mean AUC for', alg, ':', round(mean_auc[alg], 3))
	print('AUC for', alg, 'of original data set:', round(auc_org[alg], 3))
	print('')
	print('Transferability for', alg, ':', 100 - (int(np.abs(auc_org[alg] - mean_auc[alg]) * 100)), "%")
	print('Variance of', alg, 'AUC in regard to original AUC:', round(var[alg], 5))
	print('')
	print('Brier Score:', round(brier_score_loss([auc_org[alg]] * len(auc[alg]), auc[alg]), 3))
	print('')
	print('Mean absolute percentage error:',
		  round(mean_absolute_percentage_error([auc_org[alg]] * len(auc[alg]), auc[alg]), 3), '%', )
	print('\n')
