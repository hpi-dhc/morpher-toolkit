#!/usr/bin/env python
import traceback
import logging
import pandas as pd
import morpher.config as config
from morpher.config import scalers
from morpher.jobs import MorpherJob
from morpher.exceptions import kwarg_not_empty
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, QuantileTransformer 
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder
from sklearn_pandas import DataFrameMapper

class Transform(MorpherJob):

    def do_execute(self):

        filename = self.get_input("filename")
        target = self.get_input_variables("target")
        transforms = self.get_input("transforms")
        drop = self.get_input("drop")

        df = pd.read_csv(filepath_or_buffer= filename)
        df = self.execute(df, target, transform=transforms, drop=drop)
        self.add_output("filename", self.save_to_file(df))
        self.add_output("cohort_id", self.get_input("cohort_id"))
        self.add_output("user_id", self.get_input("user_id"))
        self.add_output("target", target)
        self.logger.info("Data transformed successfully.")

    def execute(self, data, transforms=None, drop=None, **kwargs):
        try:
          
          if not data.empty:

            if transforms:
                mapping = [(feature, globals()[transform_method](**kwargs) ) for feature, transform_method in transforms]
                mapper = DataFrameMapper(mapping, df_out=True, default=None)
                data = mapper.fit_transform(data.copy())
            
            if drop:
                data.drop([col for col in drop if col in data.columns], axis=1, inplace=True)

          else:
            raise AttributeError("No data provided")        

        except Exception as e:
          print(traceback.format_exc())  
          logging.error(traceback.format_exc())

        return data
