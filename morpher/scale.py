import traceback
import logging
from collections import namedtuple
from morpher.exceptions import kwarg_not_empty
import pandas as pd

from sklearn.preprocessing import (
    Normalizer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def scale(data, **kwargs):
    try:
        # TODO: test is this is null
        if not data.empty:

            scaling_class = kwargs.get("method")
            kwarg_not_empty(scaling_class, "method")
            scaler = scaling_class()

            # TODO: think about how to solve the issue of fit x fit_transform()
            scaled_df = pd.DataFrame(scaler.fit_transform(data))
            scaled_df.columns = data.columns
            scaled_df.index = data.index

            data = scaled_df

        else:
            raise AttributeError("No data provided")

    except Exception:
        logging.error(traceback.format_exc())

    return data


_options = {
    'DEFAULT': StandardScaler,
    'ROBUST': RobustScaler,
    'NORMALIZER': Normalizer,
    'QUANTILE_TRANSFORMER': QuantileTransformer
}

scaler_config = namedtuple('options', _options.keys())(**_options)
