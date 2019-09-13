from collections import namedtuple

FILE = "file"
DATABASE = "database"
DEFAULT = "default"
KNN = "knn"
ROBUST = "robust"
NORMALIZER = "normalizer"

# retaining constants for retrocompatibility
DECISION_TREE = "DecisionTree"
RANDOM_FOREST = "RandomForest"
LOGISTIC_REGRESSION = "LogisticRegression"
MULTILAYER_PERCEPTRON = "MultilayerPerceptron"
GRADIENT_BOOSTING_DECISION_TREE = "GradientBoostingDecisionTree"

# supported algorithms
algorithms = {
  'DT' : 'DecisionTree',
  'RF' : 'RandomForest',
  'LR' : 'LogisticRegression',
  'MLP': 'MultilayerPerceptron',
  'GBDT': 'GradientBoostingDecisionTree'}
  
# supported imputers
imputers = {
  'DEFAULT' : 'SimpleImputer',
  'KNN' : 'KNNImputer',
  'SOFT' : 'SoftImputer'}

algorithms, imputers = [ namedtuple('options', attr.keys())(**attr) for attr in [algorithms, imputers] ]

#algorithms = namedtuple('algorithms', algorithms.keys())(**algorithms)
#imputers = namedtuple('imputers', imputers.keys())(**imputers)
