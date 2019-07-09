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

	# get AUC results for each algorithmn
	for alg in results:

		dis[alg] = get_discrimination_metrics(**results[alg])

		auc.append(int(test['auc']*100))

		print('Original AUC:', auc_org)
		print('AUCs:', auc)

# results = pickle.load(open(r'results_performance\results.pkl', "rb"))
# auc_list = pickle.load(open(r'results_performance\auc_list.pkl', "rb"))
# auc = pickle.load(open(r'results_performance\auc.pkl', "rb"))

mean_auc = (sum(auc)+auc_org)/(len(auc)+1)

mean_auc = defaultdict(lambda: {})
for alg in results:
	mean_auc[alg] = (sum(auc[alg]))/(len(auc[alg]))

for i in auc:
	dis_to_org += (i - auc_org)**2
	dis_to_mean += (i - mean_auc)**2
r2 = 1 - (dis_to_org / dis_to_mean)

print('Distance to original dataset:', dis_to_org)
print('Distance to AUC Mean:', dis_to_mean)
print('R2:', r2)



# Output

# For each algorithmn
for alg in results:
	print('Values for', alg, ':')
	print('Mean AUC for', alg, ':', mean_auc[alg])
	print('Distance to original dataset:', dis_to_org[alg])
	print('Distance to AUC Mean:', dis_to_mean[alg])
	print('R2:', r2[alg])
	print('R2 (sklearn):',
		  r2_score([auc_org[alg]] * len(auc[alg]), auc[alg]))  # the values are extremly different
	print('Brier:', brier_score_loss([auc_org[alg]] * len(auc[alg]), auc[alg]))
	print('Mean absolute error:', mean_absolute_error([auc_org[alg]] * len(auc[alg]), auc[alg]))
	print('Mean absolute percentage error:',
		  mean_absolute_percentage_error([auc_org[alg]] * len(auc[alg]), auc[alg]), '%')
	# Percentage difference between the mean of AUCs and the original AUC
	print('Transferability for', alg, ':', 100 - (int(np.abs(auc_org[alg] - mean_auc[alg]) * 100)), "%")
	print('Variance of', alg, 'AUC in regard to orginal AUC:', var[alg])
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
for alg in results_org:
	print("Printing metrics for %s" % alg)
	print(get_calibration_metrics(**results_org[alg]))
	print(get_clinical_usefulness_metrics(get_discrimination_metrics(**results_org[alg]), tr=0.03))
	print('\n')


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
plt.figure(2, figsize=(17, 9))

for alg in results:
	plt.subplot(n)
	plt.plot(auc[alg], 'ro')
	plt.plot(auc[alg], label='AUC\'s')
	plt.plot([auc_org[alg]] * len(auc[alg]), label='AUC original dataset')
	plt.title(alg)


	plt.xlabel('Datasets')
	if n == subplot_ylabel:  # shows only the label on the left
		plt.ylabel('Area Under the ROC Curve')
	plt.title('Receiver Operating Curve')

	n += 1
#plt.subplots_adjust(left=0.1)
plt.legend(bbox_to_anchor=(1.6, 1.05))
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

# # plot Decision Curve
# plt.figure(3)
# plot_dc(results_org)
# plt.show()
#
# plot Calibration Curve
plt.figure(4)
plot_cc(models, train, test, target)
plt.legend(bbox_to_anchor=(1.3, 1.05))
plt.show()


