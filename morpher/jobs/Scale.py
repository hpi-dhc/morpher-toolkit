#!/usr/bin/env python
import traceback
import logging
import pandas as pd
import morpher.config as config
from morpher.config import scalers
from morpher.jobs import MorpherJob
from morpher.exceptions import kwarg_not_empty
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, QuantileTransformer

class Scale(MorpherJob):

    def do_execute(self):

        filename = self.get_input("filename")
        target = self.get_input_variables("target")
        transform_method = self.get_input_variables("transform_method")

        df = pd.read_csv(filepath_or_buffer= filename)
        df = self.execute(df, target, transform_method=transform_method)
        self.add_output("filename", self.save_to_file(df))
        self.add_output("cohort_id", self.get_input("cohort_id"))
        self.add_output("user_id", self.get_input("user_id"))
        self.add_output("target", target)
        self.logger.info("Data transformed successfully.")

    def execute(self, data, target, transform_method=scalers.DEFAULT,**kwargs):
        try:
          
          if not data.empty:

            transform = globals()[transform_method](**kwargs)
            labels = data[target]
            features = data.drop(target, axis=1)            
            
            ''' We refrain from transforming the label, that's why we need to concat both frames '''
            df_labels = labels.to_frame()
            df_features = pd.DataFrame(transform.fit_transform(features))
            df_labels.reset_index(drop=True, inplace=True)
            df_features.reset_index(drop=True, inplace=True)
            transformed_df = pd.concat([df_labels,df_features], axis=1)
            transformed_df.columns = [target] + list(features.columns)
            transformed_df.index = data.index
            data = transformed_df

          else:
            raise AttributeError("No data provided")        

        except Exception as e:
          print(traceback.format_exc())  
          logging.error(traceback.format_exc())

        return data       
