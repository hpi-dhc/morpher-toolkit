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

options = [
	# supported algorithms
	{'DT' : 'DecisionTree',
	 'RF' : 'RandomForest',
	 'LR' : 'LogisticRegression',
	 'MLP': 'MultilayerPerceptron',
	 'GBDT': 'GradientBoostingDecisionTree',
	 'ENLR': 'ElasticNetLR',
	 'SVM': 'SupportVectorMachine',
	 'ADABOOST': 'AdaBoost',
	 'CNBAYES': 'ComplementNaiveBayes',
	 'GNBAYES': 'GaussianNaiveBayes'},	
	# supported imputers
	{'DEFAULT' : 'SimpleImputer',
	 'KNN' : 'KNNImputer',
	 'SOFT' : 'SoftImputer'},
	# supported scalers
	{'DEFAULT' : 'StandardScaler',
	 'ROBUST' : 'RobustScaler',
	 'NORMALIZER' : 'Normalizer',
	 'QUANTILE_TRANSFORMER' : 'QuantileTransformer'},
	# supported encoders
	{'DEFAULT' : None,
	 'BINARIZER' : 'LabelBinarizer',
	 'LABEL' : 'LabelEncoder', 
	 'ONEHOT':'OneHotEncoder',
	 'ORDINAL':'OrdinalEncoder'},	
	 # supported samplers
	{'SMOTE' : 'SMOTE',	 
	 'BORDERLINE' : 'BorderlineSMOTE',
	 'ADASYN' : 'ADASYN',
	 'RANDOM' : 'RandomOverSampler',
	 'URANDOM' : 'RandomUnderSampler',
	 'CLUSTER' : 'ClusterCentroids'},
	 # supported interpreters
	{'LIME' : 'LimeExplainer',
	 'MIMIC' : 'MimicExplainer',
	 'SHAP' : 'ShapExplainer',
	 'FEAT_CONTRIB' : 'FeatContribExplainer'}]

# create the named tuples
algorithms, imputers, scalers, encoders, samplers, explainers = \
	[ namedtuple('options', attr.keys())(**attr) \
		for attr in options ]
