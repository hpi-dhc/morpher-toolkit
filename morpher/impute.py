import traceback
import logging
from morpher.exceptions import kvarg_not_empty
from morpher.imputers import *
import morpher.config as config
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

def impute(data, **kvargs):
    try:
      #TODO: test is this is null
      if not data.empty:

        method = kvargs.get("method")
        kvarg_not_empty(method,"method")

        if (method == config.DEFAULT):
          imputer = SimpleImputer()        

        elif (method == config.KNN):      
          imputer = KNNImputer()

        ''' columns where all values are NaN get assigned 0, otherwise imputer will throw them away '''
        data.loc[:, data.isna().all()] = 0.0

        imputed_df = pd.DataFrame(imputer.fit_transform(data))
        imputed_df.columns = data.columns
        imputed_df.index = data.index

        data = imputed_df

      else:
        raise AttributeError("No data provided")		

    except Exception as e:
      logging.error(traceback.format_exc())

    return data
    