import traceback
import logging
from morpher.exceptions import kvarg_not_empty
import morpher.config as config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split(data, **kvargs):

    try:
      #TODO: test is this is null
      if not data.empty:

        test_size = kvargs.get("test_size")
        kvarg_not_empty(test_size,"test_size")

        return train_test_split(data, test_size=test_size)

      else:
        raise AttributeError("No data provided")		

    except Exception as e:
      logging.error(traceback.format_exc())

    return None
