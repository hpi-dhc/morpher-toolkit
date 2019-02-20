import traceback
import logging
from morpher.exceptions import kvarg_not_empty
from morpher.imputers import *
import morpher.config as config
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer

def scale(data, **kvargs):
    try:
      #TODO: test is this is null
      if not data.empty:

        method = kvargs.get("method")
        kvarg_not_empty(method,"method")

        if (method == config.DEFAULT):
          scaler = StandardScaler()        

        elif (method == config.ROBUST):      
          scaler = RobustScaler()

        elif (method == config.NORMALIZER):      
          scaler = Normalizer()

        #TODO: think about how to solve the issue of fit x fit_transform()
        scaled_df = pd.DataFrame(scaler.fit_transform(data))
        scaled_df.columns = data.columns
        scaled_df.index = data.index

        data = scaled_df

      else:
        raise AttributeError("No data provided")		

    except Exception as e:
      logging.error(traceback.format_exc())

    return data