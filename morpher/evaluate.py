import traceback
import logging
from morpher.exceptions import kvarg_not_empty
import morpher.config as config
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error

def evaluate(data, target, algorithms, **kvargs):
	try:
		if not data.empty and algorithms and target:	    
			results = {}
			labels = data[target] #true labels
			features = data.drop(target, axis=1)
			for clf in algorithms:
				clf_name = clf.__class__.__name__
				y_true, y_pred, y_probs = labels, clf.predict(features), clf.predict_proba(features)
				results[clf_name] = { "y_true": y_true, "y_pred": y_pred, "y_probs": y_probs}
				print_clf_performance(clf_name, y_true, y_pred, y_probs)
			return results
		else:
			raise AttributeError("No data provided, algorithms or target not available")
	except Exception as e:
		logging.error(traceback.format_exc())
	return None

def print_clf_performance(clf_name, y_true, y_pred, y_probs):
	'''
	Prints performance of the prediction results
	'''
	print("***Performance report for {}".format(clf_name))

	''' report predictions '''
	print("Confusion Matrix:")
	print(confusion_matrix(y_true, y_pred))
	print("Classification report:")
	print(classification_report(y_true, y_pred))
	print("AUROC score:")
	print(roc_auc_score(y_true, y_probs[:, 1]))
	print("DOR:")
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	dor = (tp/fp)/(fn/tn)
	print(dor)
	print("***\n")

