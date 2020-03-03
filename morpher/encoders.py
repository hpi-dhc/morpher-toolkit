from collections import namedtuple

from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)

_options = {
    "DEFAULT": None,
    "BINARIZER": LabelBinarizer,
    "LABEL": LabelEncoder,
    "ONEHOT": OneHotEncoder,
    "ORDINAL": OrdinalEncoder,
}
encoder_config = namedtuple("options", _options.keys())(**_options)
