#!/usr/bin/env python
import traceback
import logging
import pandas as pd
import numpy as np
from morpher.jobs import MorpherJob
from sklearn.model_selection import train_test_split
from morpher.exceptions import kwarg_not_empty

class Split(MorpherJob):

    def do_execute(self):

        filename = self.get_input("filename")

        df = pd.read_csv(filepath_or_buffer=filename)
        test_size = self.get_input_variables("test_size")

        try:
            test_size = float(test_size)
        except ValueError:
            self.logger.error("Could not convert test_size to float, provided: {0}. Please check the parameter.".format(test_size))
            return

        train_data, test_data = self.execute(df, test_size=test_size)

        self.add_output("filenames", [self.save_to_file(train_data), self.save_to_file(test_data)])
        self.logger.info("Data split successfully, reserving {0:.0%} for test.".format(test_size))

    def execute(self, data, random_state=42, **kwargs):

        try:          
          if not data.empty:
            test_size = kwargs.get("test_size")
            kwarg_not_empty(test_size,"test_size")
            return train_test_split(data, test_size=test_size, random_state=random_state)
          else:
            raise AttributeError("No data provided")        
        except Exception as e:
          self.logger.error(traceback.format_exc())

        return None



    
