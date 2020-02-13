from collections import namedtuple

from imblearn.over_sampling import (
    ADASYN,
    BorderlineSMOTE,
    RandomOverSampler,
    SMOTE
)
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler


class NoSampler(object):
    __name__ = 'no sampler'

    def __init__(self, **kwargs):
        pass

    def fit_resample(self, X, y):
        return X, y

    def __bool(self):
        return False

    def __repr__(self):
        return self.__name__


_options = {
    'SMOTE': SMOTE,
    'BORDERLINE': BorderlineSMOTE,
    'ADASYN': ADASYN,
    'RANDOM': RandomOverSampler,
    'URANDOM': RandomUnderSampler,
    'CLUSTER': ClusterCentroids,
    'NOSAMPLER': NoSampler,
}

sampler_config = namedtuple('options', _options.keys())(**_options)
