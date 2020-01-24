from collections import namedtuple

from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
    SMOTE
)
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler

_options = {
    'SMOTE': SMOTE,
    'BORDERLINE': BorderlineSMOTE,
    'ADASYN': ADASYN,
    'RANDOM': RandomOverSampler,
    'URANDOM': RandomUnderSampler,
    'CLUSTER': ClusterCentroids
}

sampler_config = namedtuple('options', _options.keys())(**_options)
