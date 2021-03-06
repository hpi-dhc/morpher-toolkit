import os
from getpass import getpass

from morpher.algorithms import algorithm_config as algorithms
from morpher.encoders import encoder_config as encoders
from morpher.explainers import explainer_config as explainers
from morpher.imputers import imputer_config as imputers
from morpher.samplers import sampler_config as samplers
from morpher.scale import scaler_config as scalers
from morpher.selectors import selector_config as selectors

FILE = "file"
DATABASE = "database"
DEFAULT = "default"
KNN = "knn"
ROBUST = "robust"
NORMALIZER = "normalizer"

#  retaining constants for retrocompatibility
DECISION_TREE = "DecisionTree"
RANDOM_FOREST = "RandomForest"
LOGISTIC_REGRESSION = "LogisticRegression"
MULTILAYER_PERCEPTRON = "MultilayerPerceptron"
GRADIENT_BOOSTING_DECISION_TREE = "GradientBoostingDecisionTree"

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_address = os.getenv("DB_ADDRESS")
db_port = os.getenv("DB_PORT")

__all__ = [
    "algorithms",
    "imputers",
    "scalers",
    "encoders",
    "samplers",
    "explainers",
    "selectors",
]
