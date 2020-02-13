import logging
import traceback
from collections import Counter

import pandas as pd

from morpher.jobs import MorpherJob


class Sample(MorpherJob):

    def do_execute(self):

        filename = self.get_input("filename")
        target = self.get_input_variables("target")
        sampling_method = self.get_input_variables("sampling_method")

        df = pd.read_csv(filepath_or_buffer=filename)
        df = self.execute(df, target, sampling_method=sampling_method)
        self.add_output("filename", self.save_to_file(df))
        self.add_output("cohort_id", self.get_input("cohort_id"))
        self.add_output("user_id", self.get_input("user_id"))
        self.add_output("target", target)
        self.logger.info("Data sampled successfully.")

    def execute(self, data, target, sampling_method=None, **kwargs):
        try:
            if not data.empty:
                if sampling_method:
                    print(f"Performing sampling with {sampling_method.__class__.__name__} ...")

                    sampler = sampling_method(**kwargs)
                    features, labels = data.drop(target, axis=1), data[target]
                    print("Prior label distribution: ")
                    [print(f"Class:{k}/N={v}") for k, v in dict(Counter(data[target])).items()]

                    X, y = sampler.fit_resample(features, labels)
                    data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
                    data.columns = list(features.columns) + [target]

                    print("Label distribution after sampling: ")
                    [print(f"Class:{k}/N={v}") for k, v in dict(Counter(data[target])).items()]

            else:
                raise AttributeError("No data provided")

        except Exception:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return data
