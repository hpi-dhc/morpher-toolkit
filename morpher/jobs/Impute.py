#!/usr/bin/env python
import traceback
import logging
import pandas as pd
import morpher.config as config
from morpher.imputers import *
from morpher.jobs import MorpherJob
from morpher.exceptions import kwarg_not_empty
from sklearn.impute import SimpleImputer

class Impute(MorpherJob):

    def do_execute(self):

        filename = self.get_input("filename")
        df = pd.read_csv(filepath_or_buffer= filename)
        imputation_method = self.get_input_variables("imputation_method")
        df = self.execute(df, imputation_method=imputation_method)
        self.add_output("filename", self.save_to_file(df))
        self.logger.info("Data imputed successfully.")

    def execute(self, data, **kwargs):
        try:
          
          if not data.empty:

            imputation_method = kwargs.get("imputation_method")
            kwarg_not_empty(imputation_method,"imputation_method")

            if (imputation_method == config.DEFAULT):
              imputer = SimpleImputer()

            elif (imputation_method == config.KNN):      
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
          print(traceback.format_exc())  
          logging.error(traceback.format_exc())

        return data       
