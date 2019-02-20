import traceback
import logging
from morpher.exceptions import kvarg_not_empty
from morpher.imputers import *
from morpher.algorithms import *
import morpher.config as config
import pandas as pd
import numpy as np

def train(data, target, **kvargs):
    try:
      #TODO: test is this is null
      if not data.empty:

        ''' if split_data was called beforehand, data contains a subset of the original available data '''
        labels = data[target]
        features = data.drop(target, axis=1)        

        algorithms = kvargs.get("algorithms")
        kvarg_not_empty(algorithms,"algorithms")

        trained_algs = []

        for algorithm in algorithms:
        	clf = eval("{algorithm}()".format(algorithm=algorithm)) #instantiate algorithm in runtime
        	clf.fit(features, labels)
        	trained_algs.append(clf)

        return trained_algs

      else:
        raise AttributeError("No data provided")		

    except Exception as e:
      logging.error(traceback.format_exc())

    return data
