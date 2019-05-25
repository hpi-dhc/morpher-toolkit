from datetime import datetime

from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, brier_score_loss, explained_variance_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import BayesianRidge
import sklearn.linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, f_classif
from sklearn.exceptions import NotFittedError

import pandas as pd
import numpy as np
import morpher.config

import traceback
import logging

class Base:

    def __init__(self, clf=None, optimize=None, crossval=None):

        '''
        Base classifier
        '''
        self.clf = clf

        '''
        Indicates whether we should perform gridsearch
        '''
        self.optimize = optimize

        '''
        Indicates whether we should cross-validate
        '''
        self.crossval = crossval

    def fit(self, features, labels):

        '''
        Fits a given algorithm, optionally performing hyper parameter tuning.
        '''
        
        try:
            start = datetime.now()

            if not self.optimize:
                logging.info("*** Training of model '{0}' started.".format(self.clf.__class__.__name__))            
                self.clf.fit(features, labels)              

            else:
                logging.info("Starting gridsearch for {}...".format(self.clf.estimator.__class__.__name__))
                self.clf.fit(np.asarray(features), np.asarray(labels))
                print("Came up with params: {0}".format(self.clf.best_params_))
                print("Achieved AUROC of {0}".format(self.clf.best_score_))
                self.clf = clf.best_estimator_

            end = datetime.now()
            logging.info("*** Training of classifier ready. Time elapsed: {}ms".format((end - start).microseconds/1000))

            if self.crossval:

                ''' performing cross validation '''
                logging.info("*** Performing cross validation of classifier.")

                ''' keeps class proportion balanced across folds '''
                skf = StratifiedKFold(n_splits=10)

                ''' performs cross validation and stores the results in the respective variables '''
                y_true, y_pred, y_probs = labels, cross_val_predict(self.clf, features, labels, cv=skf),\
                                                cross_val_predict(self.clf, features, labels, cv=skf, method='predict_proba')

                ''' performing cross validation '''
                logging.info("Model cross-validation performed for {0}.".format(self.clf.__class__.__name__))
        
        except Exception as e:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return None

    def predict(self, features):
        '''
        Provides the algorithm predictions for a given set of features and labels 
        '''
        try:
            return self.clf.predict(features)
        except Exception as e:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return None

    def predict_proba(self, features):
        '''
        Provides the algorithm probability predictions for a given set of features 
        '''
        try:
            return self.clf.predict_proba(features)

        except Exception as e:
            logging.error(traceback.format_exc())

        return None

    def get_params(self):
        '''
        Provides the algorithm's hyperparameter list 
        '''
        try:
            return self.clf.get_params()

        except Exception as e:
            logging.error(traceback.format_exc())

        return None

class DecisionTree(Base):

    def __init__(self, optimize=None, crossval=None):

        if not optimize:
                clf = DecisionTreeClassifier(
                max_depth = 5,
                class_weight={0:1, 1:10}
            )
        else:
            ''' gridsearch '''
            param_grid = {
                "max_depth": range(4,7),
                "min_samples_split": range(3,7),
                "min_samples_leaf": range(1, 16)
                #"min_impurity_decrease": np.arange(0.0, 0.3, 0.025)
            }
            clf = GridSearchCV(
                estimator = DecisionTreeClassifier(),
                cv = 5,
                n_jobs = -1,
                scoring = roc_auc_score,
                param_grid = param_grid
            )
        
        super().__init__(clf, optimize, crossval)

class RandomForest(Base):

    def __init__(self, optimize=None, crossval=None):

        if not optimize:
            '''
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            '''
            n_estimators = 300
            max_depth = 16
            max_features = 8
            min_samples_leaf = 4
            min_samples_split = 8
            n_estimators = 300

            clf = RandomForestClassifier(
                n_estimators = n_estimators,
                max_depth = max_depth,
                max_features = max_features,
                min_samples_leaf = min_samples_leaf,
                min_samples_split = min_samples_split
            )
        else:
            ''' gridsearch '''
            param_grid = {

                'bootstrap': [True],
                'max_depth': [2, 4, 8, 16],
                'max_features': [2, 4, 8],
                'min_samples_leaf': [3, 4, 5],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [100, 200, 300]
            }
            clf = GridSearchCV(
                estimator = RandomForestClassifier(),
                cv = 5,
                n_jobs = -1,
                scoring = roc_auc_score,
                param_grid = param_grid
            )
        
        super().__init__(clf, optimize, crossval)

class MultilayerPerceptron(Base):

    def __init__(self, optimize=None, crossval=None):

        if not optimize:
            '''
            Trains and stores a multilayer perceptron classifier on the
            current data using the current pipeline.
            '''
            activation='tanh'
            solver='sgd'
            alpha=1e-5
            hidden_layer_sizes=(21, 2)
            max_iter=500

            clf = MLPClassifier(
                activation=activation,
                solver=solver,
                alpha=alpha,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter
            )
        else:
            ''' gridsearch '''
            param_grid = [
                {
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'hidden_layer_sizes': [
                        (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,),
                        (16,), (17,), (18,), (19,), (20,), (21,)
                    ]
                }
            ]
            clf = GridSearchCV(
                estimator = MLPClassifier(),
                cv = 5,
                n_jobs = -1,
                scoring = roc_auc_score,
                param_grid = param_grid
            )
        
        super().__init__(clf, optimize, crossval)

class GradientBoostingDecisionTree(Base):

    def __init__(self, optimize=None, crossval=None):

        if not optimize:
            '''
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            '''
            learning_rate = 0.1
            n_estimators = 150
            max_depth = 3

            clf = GradientBoostingClassifier(
                learning_rate = learning_rate,
                n_estimators = n_estimators,
                max_depth = max_depth
            )
        else:
            ''' gridsearch '''
            param_grid = {
                #"learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                "max_depth": range(2,4),
                "n_estimators": range(100, 200, 25)
            }
            clf = GridSearchCV(
                estimator = GradientBoostingClassifier(),
                cv = 5,
                n_jobs = -1,
                scoring = roc_auc_score,
                param_grid = param_grid
            )
        
        super().__init__(clf, optimize, crossval)

class LogisticRegression(Base):

    def __init__(self, optimize=None, crossval=None):

        if not optimize:
            '''
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            '''
            clf = sklearn.linear_model.LogisticRegression(
                penalty = 'l1',
                C = 1.0                
            )
        else:
            ''' gridsearch '''
            param_grid = {
                "penalty": ['l1', 'l2'],
                "C": np.logspace(0, 4, 10)

            }
            clf = GridSearchCV(
                estimator = LogisticRegression(),
                cv = 5,
                n_jobs = -1,
                scoring = roc_auc_score,
                param_grid = param_grid
            )
        
        super().__init__(clf, optimize, crossval)




        









