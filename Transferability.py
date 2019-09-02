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
from stroke_modeling import *

# ToDo change structure
# ToDo get calibration metrics and build score
# ToDo get net benefit metrics and build score
# ToDo build a combination of different scores


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ---------------------------------------------------------------------------------------------------------------------
# Getting models

data = Load().execute(source=config.FILE, filename='stroke_preprocessed_imputed_lvef.csv')
data = Impute().execute(data, imputation_method=config.DEFAULT)

#train, test, models, results_org, auc_org_list, auc_org = train_model(data)

train = pickle.load(open(r'results_performance\train.pkl', "rb"))
test = pickle.load(open(r'results_performance\test.pkl', "rb"))
models = pickle.load(open(r'results_performance\models.pkl', "rb"))
results_org = pickle.load(open(r'results_performance\results_org.pkl', "rb"))
auc_org_list = pickle.load(open(r'results_performance\auc_list.pkl', "rb"))
auc_org = pickle.load(open(r'results_performance\auc_org.pkl', "rb"))
a_cal_org_list = pickle.load(open(r'results_performance\a_cal_org_list.pkl', "rb"))
a_cal_org = pickle.load(open(r'results_performance\a_cal_org.pkl', "rb"))
b_cal_org_list = pickle.load(open(r'results_performance\b_cal_org_list.pkl', "rb"))
b_cal_org = pickle.load(open(r'results_performance\b_cal_org.pkl', "rb"))

results = pickle.load(open(r'results_performance\results.pkl', "rb"))
auc_list = pickle.load(open(r'results_performance\auc_list.pkl', "rb"))
auc = pickle.load(open(r'results_performance\auc.pkl', "rb"))
a_cal_list = pickle.load(open(r'results_performance\a_cal_list.pkl', "rb"))
a_cal = pickle.load(open(r'results_performance\a_cal.pkl', "rb"))
b_cal_list = pickle.load(open(r'results_performance\b_cal_list.pkl', "rb"))
b_cal = pickle.load(open(r'results_performance\b_cal.pkl', "rb"))

# ----------------------------------------------------------------------------------------------------------------------
# mean calculations
mean_auc_total = (sum(auc_list))/(len(auc_list)) # mean without original AUC
mean_auc_org_total = (sum(auc_org_list))/(len(auc_org_list)) # calculates an average as AUC original score of all algorithmns

mean_a_cal_total = (sum(a_cal_list))/(len(a_cal_list))
mean_a_cal_org_total = (sum(a_cal_org_list))/(len(a_cal_org_list))

mean_b_cal_total = (sum(b_cal_list))/(len(b_cal_list))
mean_b_cal_org_total = (sum(b_cal_org_list))/(len(b_cal_org_list))

mean_auc = defaultdict(lambda: {})
for alg in results:
	mean_auc[alg] = (sum(auc[alg]))/(len(auc[alg]))

mean_a_cal = defaultdict(lambda: {})
for alg in results:
	mean_a_cal[alg] = (sum(a_cal[alg]))/(len(a_cal[alg]))

mean_b_cal = defaultdict(lambda: {})
for alg in results:
	mean_b_cal[alg] = (sum(b_cal[alg]))/(len(b_cal[alg]))

# distance to original and mean AUC and R2

# for total AUC
dis_to_auc_org_total = 0
dis_to_mean_auc_total = 0
r2_total = 0

dis_to_a_cal_org_total = 0
dis_to_mean_a_cal_total = 0

dis_to_b_cal_org_total = 0
dis_to_mean_b_cal_total = 0

for auc_i in auc_list:
	dis_to_auc_org_total += (auc_i - mean_auc_org_total)**2
	#dis_to_mean_total += (auc_org - mean_auc)**2 ?? Harry why did you use the original auc here??
	dis_to_mean_auc_total += (auc_i - mean_auc_total)**2
r2_total = 1 - (dis_to_auc_org_total / dis_to_mean_auc_total)

for a_cal_i in a_cal_list:
	dis_to_a_cal_org_total += (a_cal_i - mean_a_cal_org_total)**2
	dis_to_mean_a_cal_total += (a_cal_i - mean_a_cal_total)**2

for b_cal_i in b_cal_list:
	dis_to_b_cal_org_total += (b_cal_i - mean_b_cal_org_total)**2
	dis_to_mean_b_cal_total += (b_cal_i - mean_b_cal_total)**2


# for each algorithmn
dis_auc_mean = 0
dis_auc_org = 0
dis_to_auc_org = defaultdict(lambda: {})
dis_to_auc_mean = defaultdict(lambda: {})
#r2 = defaultdict(lambda: {})

dis_a_cal_mean = 0
dis_a_cal_org = 0
dis_to_a_cal_org = defaultdict(lambda: {})
dis_to_a_cal_mean = defaultdict(lambda: {})

dis_b_cal_mean = 0
dis_b_cal_org = 0
dis_to_b_cal_org = defaultdict(lambda: {})
dis_to_b_cal_mean = defaultdict(lambda: {})

for alg in results:
	for auc_i in auc[alg]:
		dis_auc_org += (auc_i - auc_org[alg]) ** 2
		dis_auc_mean += (auc_i - mean_auc[alg]) ** 2
	dis_to_auc_org[alg] = dis_auc_org
	dis_to_auc_mean[alg] = dis_auc_mean
	#r2[alg] = 1 - (dis_to_auc_org[alg] / dis_to_auc_mean[alg])

for alg in results:
	for a_cal_i in a_cal[alg]:
		dis_a_cal_org += (a_cal_i - a_cal_org[alg]) ** 2
		dis_a_cal_mean += (a_cal_i - mean_a_cal[alg]) ** 2
	dis_to_a_cal_org[alg] = dis_a_cal_org
	dis_to_a_cal_mean[alg] = dis_a_cal_mean

for alg in results:
	for b_cal_i in b_cal[alg]:
		dis_b_cal_org += (b_cal_i - b_cal_org[alg]) ** 2
		dis_b_cal_mean += (b_cal_i - mean_b_cal[alg]) ** 2
	dis_to_b_cal_org[alg] = dis_b_cal_org
	dis_to_b_cal_mean[alg] = dis_b_cal_mean


# variance regarding original AUC
var_auc = defaultdict(lambda: {})
var_auc_total = 0
for alg in results:
	var_auc[alg] = sum((i - auc_org[alg]) ** 2 for i in auc[alg]) / len(auc[alg])
var_auc_total = sum((i - mean_auc_org_total) ** 2 for i in auc_list) / len(auc_list)

var_a_cal = defaultdict(lambda: {})
var_a_cal_total = 0
for alg in results:
	var_a_cal[alg] = sum((i - a_cal_org[alg]) ** 2 for i in a_cal[alg]) / len(a_cal[alg])
var_a_cal_total = sum((i - mean_a_cal_org_total) ** 2 for i in a_cal_list) / len(a_cal_list)

var_b_cal = defaultdict(lambda: {})
var_b_cal_total = 0
for alg in results:
	var_b_cal[alg] = sum((i - b_cal_org[alg]) ** 2 for i in b_cal[alg]) / len(b_cal[alg])
var_b_cal_total = sum((i - mean_b_cal_org_total) ** 2 for i in b_cal_list) / len(b_cal_list)

# ----------------------------------------------------------------------------------------------------------------------

# Output

# For each algorithmn
# for alg in results:
# 	print('Transferability Performances Measures for', alg, ':')
# 	print('\n')
# 	print('Mean AUC for', alg, ':', round(mean_auc[alg], 3))
# 	print('Distance to original dataset:', round(dis_to_org[alg], 3))
# 	print('Distance to AUC Mean:', round(dis_to_mean[alg], 3))
# 	#print('R2:', round(r2[alg], 3))
# 	print('R2 (sklearn):',
# 		  round(r2_score([auc_org[alg]] * len(auc[alg]), auc[alg]), 2))  # the values are extremly different
# 	print('Brier:', round(brier_score_loss([auc_org[alg]] * len(auc[alg]), auc[alg]), 3))
# 	print('Mean absolute error:', round(mean_absolute_error([auc_org[alg]] * len(auc[alg]), auc[alg]), 3))
# 	print('Mean absolute percentage error:',
# 		  round(mean_absolute_percentage_error([auc_org[alg]] * len(auc[alg]), auc[alg]), 3), '%', )
# 	# Percentage difference between the mean of AUCs and the original AUC
# 	print('Transferability for', alg, ':', 100 - (int(np.abs(auc_org[alg] - mean_auc[alg]) * 100)), "%")
# 	print('Variance of', alg, 'AUC in regard to orginal AUC:', round(var[alg], 5))
# 	print('\n')
#
# # Total Scores
# print('Total Scores:')
# print('Mean AUC:', mean_auc_total)
# print('Distance to original dataset:', dis_to_org_total)
# print('Distance to AUC Mean:', dis_to_mean_total)
# print('R2:', r2_total)
# print('R2 (sklearn):', r2_score([mean_auc_org_total] * len(auc_list), auc_list)) # the values are extremly different, often zero
# print('Brier:', brier_score_loss([mean_auc_org_total] * len(auc_list), auc_list))
# print('Mean absolute error:', mean_absolute_error([mean_auc_org_total] * len(auc_list), auc_list))
# print('Mean absolute percentage error:', mean_absolute_percentage_error([mean_auc_org_total] * len(auc_list), auc_list), '%')
# # Percentage difference between the mean of AUCs and the original AUC
# print('Transferability_overall:', 100 - (int(np.abs(mean_auc_org_total - mean_auc_total) * 100)), "%")
# print('Variance of AUC in regard to orginal AUC:', var_total)
# print('\n')
#
# # Calibrations and Clinical Usefulness metrics
# for alg in results_org:
# 	print("Printing metrics for %s" % alg)
# 	print(get_discrimination_metrics(**results_org[alg]))
# 	print(get_calibration_metrics(results_org[alg]["y_true"], results_org[alg]["y_probs"]))
# 	print(get_clinical_usefulness_metrics(get_discrimination_metrics(**results_org[alg]), tr=0.03))
# 	print('\n')


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
# n = 101  # subplot number must be a three digit number
# n += len(results) * 10  # second digit shows number of graph in one row
# subplot_ylabel = n
# plt.figure(2, figsize=(18.5, 7))
#
# for alg in results:
# 	plt.subplot(n)
# 	plt.plot(auc[alg], 'ro')
# 	plt.plot(auc[alg], label='AUC\'s')
# 	plt.plot([auc_org[alg]] * len(auc[alg]), label='AUC original dataset')
# 	plt.title('Area Under the ROC Curv')
#
#
# 	plt.xlabel('Datasets')
# 	if n == subplot_ylabel:  # shows only the label on the left
# 		plt.ylabel(alg)
# 	plt.title(alg)
#
# 	n += 1
# #plt.subplots_adjust(left=0.1)
# plt.legend(bbox_to_anchor=(0.8, 1.05))
# plt.show()

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
	print('Transferability for', alg, ':', 100 - (int(np.abs(auc_org[alg] - mean_auc[alg]) * 100)), "%")
	print('Variance of', alg, 'AUC in regard to original AUC:', round(var_auc[alg], 5))
	print('Brier Score:', round(brier_score_loss([auc_org[alg]] * len(auc[alg]), auc[alg]), 3))
	print('Mean absolute percentage error:',
		  round(mean_absolute_percentage_error([auc_org[alg]] * len(auc[alg]), auc[alg]), 3), '%', )
	print('\n')
	print('Mean A_cal for', alg, ':', round(mean_a_cal[alg], 3))
	print('A_cal for', alg, 'of original data set:', round(a_cal_org[alg], 3))
	print('Transferability for', alg, ':', 100 - (int(np.abs(a_cal_org[alg] - mean_a_cal[alg]) * 100)), "%")
	print('Variance of', alg, 'A_cal in regard to original A_cal:', round(var_a_cal[alg], 5))
	print('Brier Score:', round(brier_score_loss([a_cal_org[alg]] * len(a_cal[alg]), a_cal[alg]), 3))
	print('Mean absolute percentage error:',
		  round(mean_absolute_percentage_error([a_cal_org[alg]] * len(a_cal[alg]), a_cal[alg]), 3), '%', )
	print('\n')
	print('Mean B_cal for', alg, ':', round(mean_b_cal[alg], 3))
	print('B_cal for', alg, 'of original data set:', round(b_cal_org[alg], 3))
	print('Transferability for', alg, ':', 100 - (int(np.abs(b_cal_org[alg] - mean_b_cal[alg]) * 100)), "%")
	print('Variance of', alg, 'B_cal in regard to original B_cal:', round(var_b_cal[alg], 5))
	if b_cal[alg][0] < 1:
		print('Brier Score:', round(brier_score_loss([b_cal_org[alg]] * len(b_cal[alg]), b_cal[alg]),
									3))  # B_cal can be > 1, therefore Brier Score can not always be applied
	print('Mean absolute percentage error:',
		  round(mean_absolute_percentage_error([a_cal_org[alg]] * len(b_cal[alg]), b_cal[alg]), 3), '%', )
	print('\n')
