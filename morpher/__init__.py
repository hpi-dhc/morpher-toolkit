import logging
import morpher.config
import morpher.jobs
import numpy as np

#Setting random seed to be picked up by frameworks, such as sklearn
np.random.seed(1729)

logging.basicConfig(filename='morpher.log',level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')