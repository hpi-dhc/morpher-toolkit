import traceback
import logging
from collections import namedtuple
from datetime import datetime

import lightgbm
import numpy as np
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression as sklearnLogisticRegression,
    SGDClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    StratifiedKFold,
)
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from morpher.metrics import get_discrimination_metrics


class Base:
    def __init__(
        self,
        clf=None,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        n_jobs=None,
        verbose=False,

    ):

        """
        Base classifier
        """
        self.clf = clf

        """
        Indicates whether we should perform gridsearch
        """
        self.optimize = optimize

        """
        Indicates whether we should cross-validate
        """
        self.crossval = crossval

        """
        Indicates how many folds to cross-validate
        """
        self.n_splits = n_splits

        """
        Indicates how many jobs for fit should run in parallel
        """
        self.n_jobs = n_jobs

        """
        Verbose mode will print statistics to prompt while training
        """
        self.verbose = verbose

        """
        Keeps track of the classes used in the classifier, here for compatibility with sklearn and others
        """
        self.classes_ = None


    def fit(self, features, labels):

        """
        Fits a given algorithm, optionally performing hyperparameter tuning.
        """

        try:
            start = datetime.now()

            if not self.optimize:
                msg = "*** Training of model '{0}' started.".format(
                    self.clf.__class__.__name__
                )
                logging.info(msg)

                if self.verbose:
                    print(msg)
                
                self.clf.fit(features, labels)

            else:
                msg = "Starting gridsearch for {}...".format(
                    self.clf.estimator.__class__.__name__
                )
                logging.info(msg)
                
                if self.verbose:
                    print(msg)
                
                self.clf.fit(np.asarray(features), np.asarray(labels))
                
                if self.verbose:
                    print("Came up with params: {0}".format(self.clf.best_params_))
                    print("Achieved AUROC of {0}\n".format(self.clf.best_score_))
                
                """ clf now becomes the best estimator """
                self.clf = self.clf.best_estimator_

            end = datetime.now()
            msg = "*** Training of classifier ready. Time elapsed: {}ms\n".format(
                (end - start).microseconds / 1000
            )
            self.classes_ = self.clf.classes_
            
            logging.info(msg)

            if self.verbose:
                print(msg)

            if self.crossval:

                """ performing cross validation """
                msg = "*** Performing cross validation of '{0}' classifier.".format(
                    self.clf.__class__.__name__
                )
                logging.info(msg)
                if self.verbose:
                    print(msg)
                """ keeps class proportion balanced across folds """
                n_splits = self.n_splits or 10
                print(f"Number of splits: {n_splits}")
                skf = StratifiedKFold(n_splits=n_splits)

                """ performs cross validation and stores the results in the respective variables """
                n_jobs = self.n_jobs or 1
                y_true, y_pred, y_probs = (
                    labels,
                    cross_val_predict(
                        self.clf,
                        features,
                        labels,
                        cv=skf,
                        n_jobs=n_jobs
                    ),
                    cross_val_predict(
                        self.clf,
                        features,
                        labels,
                        cv=skf,
                        method="predict_proba",
                        n_jobs=n_jobs
                    ),
                )

                """ performing cross validation """
                msg = "Model cross-validation performed for {0}.".format(
                    self.clf.__class__.__name__
                )

                logging.info(msg)
                if self.verbose:
                    print(msg)

                """ if cross validation was required, return the discrimination metrics from crossvalidation """
                """ note slicing of y_probs, we do it to get the prediction for label = 1"""
                return get_discrimination_metrics(
                    y_true, y_pred, y_probs[:, 1]
                )

        except Exception:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return None

    def predict(self, features):
        """
        Provides the algorithm predictions for a given set of features and labels
        """
        try:
            return self.clf.predict(features)
        except Exception:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return None

    def predict_proba(self, features):
        """
        Provides the algorithm probability predictions for a given set of features
        """
        try:
            return self.clf.predict_proba(features)

        except Exception:
            logging.error(traceback.format_exc())

        return None

    def get_params(self, deep=True):
        """
        Provides the algorithm's hyperparameter list
        """
        try:
            return self.clf.get_params(deep)

        except Exception:
            logging.error(traceback.format_exc())

        return None

    def score_auroc(self, estimator, X, y):
        """
        AUROC scoring method for estimators (classifiers) so we can use AUROC
        for model evaluation during hyperparam optimization using GridSearch.
        """
        y_pred = estimator.predict_proba(X)
        return roc_auc_score(y, y_pred[:, 1])

    @property
    def is_tree_(self):
        """
        Property used to define, for example, if we should use SHAP Tree Explainer instead of KernelExplainer.
        """
        return False


    def set_params(self, **kwargs):
        self.clf.set_params(**kwargs)
        return self

class Dummy(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
    ):
        if not hyperparams:
            hyperparams = {"strategy": "most_frequent"}
        if not optimize:
            clf = DummyClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {}
            clf = GridSearchCV(
                estimator=DummyClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )


class DecisionTree(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:
            hyperparams = {"max_depth": 5, "class_weight": "balanced"}

        if not optimize:
            if kwargs:
                clf = DecisionTreeClassifier(**kwargs)
            else:
                clf = DecisionTreeClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "max_depth": range(4, 7),
                    "min_samples_split": range(3, 7),
                    "min_samples_leaf": range(1, 16),
                    "min_impurity_decrease": np.arange(0.0, 0.3, 0.025),
                }
            clf = GridSearchCV(
                estimator=DecisionTreeClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    """
    Stores model-based feature importance, for models such as trees
    """

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "feature_importances_"):
            return self.clf.feature_importances_
        else:
            return False


class RandomForest(Base):
    def __init__(
        self,
        hyperparams={},
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs,
    ):
        if not hyperparams:

            hyperparams = {
                "n_estimators": 300,
                "max_depth": 16,
                "min_samples_leaf": 4,
                "min_samples_split": 8,
                "n_estimators": 300,
                "class_weight": "balanced",
            }

        if not optimize:
            """
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = RandomForestClassifier(**kwargs)
            else:
                clf = RandomForestClassifier(**hyperparams)

        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "bootstrap": [True],
                    "max_depth": [2, 4, 8, 16],
                    "max_features": [2, 4, 8],
                    "min_samples_leaf": [3, 4, 5],
                    "min_samples_split": [8, 10, 12],
                    "n_estimators": [100, 200, 300],
                    "class_weight": ["balanced"],
                }
            clf = GridSearchCV(
                estimator=RandomForestClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    """
    Stores model-based feature importance, for models such as trees
    """

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "feature_importances_"):
            return self.clf.feature_importances_
        else:
            return False

    @property
    def is_tree_(self):
        return True


class MultilayerPerceptron(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):
        if not hyperparams:

            hyperparams = {
                "activation": "tanh",
                "solver": "sgd",
                "alpha": 1e-5,
                "hidden_layer_sizes": (21, 2),
                "max_iter": 500,
            }

        if not optimize:
            """
            Trains and stores a multilayer perceptron classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = MLPClassifier(**kwargs)
            else:
                clf = MLPClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = [
                    {
                        "activation": ["identity", "logistic", "tanh", "relu"],
                        "solver": ["lbfgs", "sgd", "adam"],
                        "hidden_layer_sizes": [
                            (1,),
                            (2,),
                            (3,),
                            (4,),
                            (5,),
                            (6,),
                            (7,),
                            (8,),
                            (9,),
                            (10,),
                            (11,),
                            (12,),
                            (13,),
                            (14,),
                            (15,),
                            (16,),
                            (17,),
                            (18,),
                            (19,),
                            (20,),
                            (21,),
                        ],
                    }
                ]
            clf = GridSearchCV(
                estimator=MLPClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )


class GradientBoostingDecisionTree(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:
            hyperparams = {
                "learning_rate": 0.1,
                "n_estimators": 150,
                "max_depth": 3,
            }

        if not optimize:
            """
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = GradientBoostingClassifier(**kwargs)
            else:
                clf = GradientBoostingClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "learning_rate": [
                        0.001,
                        0.005,
                        0.01,
                        0.05,
                        0.1,
                        0.2,
                        0.5,
                        1,
                    ],
                    "max_depth": range(2, 10),
                    "n_estimators": range(100, 200, 25),
                }
            clf = GridSearchCV(
                estimator=GradientBoostingClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
                verbose=1
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "feature_importances_"):
            return self.clf.feature_importances_
        else:
            return False

    @property
    def is_tree_(self):
        return True


class XGBoost(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:
            hyperparams = {"objective": "binary:logistic"}

        if not optimize:
            """
            Trains and stores a XGBoost classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = xgb.XGBClassifier(**kwargs)
            else:
                clf = xgb.XGBClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "n_estimators": [50, 100, 150, 200],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "max_depth": range(3, 10),
                    "colsample_bytree": [i/10.0 for i in range(1, 3)],
                    "gamma": [i/10.0 for i in range(3)],
                }
            clf = GridSearchCV(
                estimator=xgb.XGBClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_

    @property
    def is_tree_(self):
        return True


class LightGBM(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:
            hyperparams = {}

        if not optimize:
            """
            Trains and stores a XGBoost classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = lightgbm.LGBMClassifier(**kwargs)
            else:
                clf = lightgbm.LGBMClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "n_estimators": [50, 100, 150, 200],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3],
                    "max_depth": range(3, 10),
                    "num_leaves": [i * 10.0 for i in range(10)],
                }
            clf = GridSearchCV(
                estimator=lightgbm.LGBMClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_

    @property
    def is_tree_(self):
        return True


class AdaBoost(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:
            hyperparams = {"learning_rate": 1, "n_estimators": 150}

        if not optimize:
            """
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = AdaBoostClassifier(**kwargs)
            else:
                clf = AdaBoostClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "learning_rate": [
                        0.001,
                        0.005,
                        0.01,
                        0.05,
                        0.1,
                        0.2,
                        0.5,
                        1,
                    ],
                    "n_estimators": range(100, 200, 25),
                }
            clf = GridSearchCV(
                estimator=AdaBoostClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "feature_importances_"):
            return self.clf.feature_importances_
        else:
            return False

    @property
    def is_tree_(self):
        return False


class LogisticRegression(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:

            hyperparams = {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "class_weight": "balanced",
            }

        if not optimize:
            """
            Trains and stores a random forest classifier on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = sklearnLogisticRegression(**kwargs)
            else:
                clf = sklearnLogisticRegression(**hyperparams)

        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "penalty": ["l2"],
                    "C": np.logspace(0, 4, 10),
                    "solver": ["lbfgs"],
                    "class_weight": ["balanced"],
                }
            clf = GridSearchCV(
                estimator=sklearnLogisticRegression(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "coef_"):
            return self.clf.coef_[0]
        else:
            return False


class SupportVectorMachine(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:

            hyperparams = {
                "probability": True,
                "C": 1.0,
                "kernel": "linear",
                "class_weight": "balanced",  # penalize
            }

        if not optimize:
            """
            Trains and stores a support vector machine on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = SVC(**kwargs)
            else:
                clf = SVC(**hyperparams)

        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "probability": [True],
                    "C": np.logspace(0, 4, 10),
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "class_weight": ["balanced"],  # penalize
                }
            clf = GridSearchCV(
                estimator=SVC(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "coef_"):
            return self.clf.coef_[0]
        else:
            return False


class ElasticNetLR(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:

            hyperparams = {"loss": "log", "penalty": "elasticnet"}

        if not optimize:
            """
            Trains and stores a logistic regression classifier using elastic net on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = SGDClassifier(**kwargs)
            else:
                clf = SGDClassifier(**hyperparams)
        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "loss": ["log"],
                    "penalty": ["elasticnet"],
                    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
                    "l1_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                }
            clf = GridSearchCV(
                estimator=SGDClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

    @property
    def feature_importances_(self):
        if hasattr(self.clf, "coef_"):
            return self.clf.coef_[0]
        else:
            return False


class ComplementNaiveBayes(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:

            hyperparams = {"alpha": 1.0, "fit_prior": True}

        if not optimize:
            """
            Trains and stores a logistic regression classifier using elastic net on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = ComplementNB(**kwargs)
            else:
                clf = ComplementNB(**hyperparams)

        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
                    "fit_prior": [True],
                }
            clf = GridSearchCV(
                estimator=ComplementNB(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )


class GaussianNaiveBayes(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        if not hyperparams:
            hyperparams = {}

        if not optimize:
            """
            Trains and stores a logistic regression classifier using elastic net on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = GaussianNB(**kwargs)
            else:
                clf = GaussianNB(**hyperparams)

        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {}
            clf = GridSearchCV(
                estimator=GaussianNB(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

class CatBoost(Base):
    def __init__(
        self,
        hyperparams=None,
        optimize=None,
        param_grid=None,
        crossval=None,
        n_splits=None,
        verbose=None,
        **kwargs
    ):

        from catboost import CatBoostClassifier 

        if not hyperparams:
            hyperparams = {}

        if not optimize:
            """
            Trains and stores a cat boost classifier using elastic net on the
            current data using the current pipeline.
            """
            if kwargs:
                clf = CatBoostClassifier(**kwargs)
            else:
                clf = CatBoostClassifier(**hyperparams)

        else:
            """ gridsearch """
            if not param_grid:
                param_grid = {}
            clf = GridSearchCV(
                estimator=CatBoostClassifier(**hyperparams),
                cv=5,
                n_jobs=-1,
                scoring=self.score_auroc,
                param_grid=param_grid,
            )

        super().__init__(
            clf, hyperparams, optimize, param_grid, crossval, n_splits
        )

_options = {
    "DUMMY": Dummy,
    "DT": DecisionTree,
    "RF": RandomForest,
    "LR": LogisticRegression,
    "MLP": MultilayerPerceptron,
    "GBDT": GradientBoostingDecisionTree,
    "ENLR": ElasticNetLR,
    "SVM": SupportVectorMachine,
    "ADABOOST": AdaBoost,
    "CNBAYES": ComplementNaiveBayes,
    "GNBAYES": GaussianNaiveBayes,
    "XGBOOST": XGBoost,
    "LIGHTGBM": LightGBM,
    "CATBOOST": CatBoost,

}

algorithm_config = namedtuple("options", _options.keys())(**_options)
