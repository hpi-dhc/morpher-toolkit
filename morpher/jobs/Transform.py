import logging
import traceback

import pandas as pd
from sklearn_pandas import DataFrameMapper

from morpher.jobs import MorpherJob


class Transform(MorpherJob):
    def do_execute(self):

        filename = self.get_input("filename")
        target = self.get_input_variables("target")
        transforms = self.get_input("transforms")
        drop = self.get_input("drop")

        df = pd.read_csv(filepath_or_buffer=filename)
        df = self.execute(df, target, transform=transforms, drop=drop)
        self.add_output("filename", self.save_to_file(df))
        self.add_output("cohort_id", self.get_input("cohort_id"))
        self.add_output("user_id", self.get_input("user_id"))
        self.add_output("target", target)
        self.logger.info("Data transformed successfully.")

    def execute(self, data, transforms=None, target=None, mapper=None, drop=None, default=None, **kwargs):
        try:

            if not data.empty:

                features, labels = data, None

                if mapper is None:

                    mapping = []
                    if transforms:
                        mapping = [
                            (feature, transform_method(**kwargs))
                            for feature, transform_method in transforms
                        ]

                    mapper = DataFrameMapper(
                        mapping, df_out=True, default=default
                    )

                    if target is not None:
                        features, labels = data.drop(target), data[target]

                    if labels is None:
                        features = mapper.fit_transform(features.copy())
                    else:
                        features = mapper.fit_transform(features.copy(), labels.copy())
                else:
                    features = mapper.transform(features.copy())

                if drop:

                    features.drop(
                        [col for col in drop if col in features.columns],
                        axis=1,
                        inplace=True
                    )

                data = features
                
                if labels is not None:
                    data[target] = labels
                    
            else:
                raise AttributeError("No data provided")

        except Exception:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return data, mapper
