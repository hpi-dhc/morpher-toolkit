import logging
import traceback
import pandas as pd
from morpher.jobs import MorpherJob

"""
Select Job makes it possible to access a range of feature selection techniques with a unified interface
Upon calling .execute() you can specify the feature selection method and how many features shall be returned.
"""
class Select(MorpherJob):
    def do_execute(self):

        filename = self.get_input("filename")

        df = pd.read_csv(filepath_or_buffer=filename)
        selection_method = self.get_input_variables("selection_method")
        df, imputer = self.execute(df, selection_method=selection_method)
        self.add_output("filename", self.save_to_file(df))
        self.add_output("cohort_id", self.get_input("cohort_id"))
        self.add_output("user_id", self.get_input("user_id"))
        self.logger.info("Features selected successfully.")

    
    """
    Selects features in a given dataset using a number of predefined feature selection techniques

    Params:

    data                    the data which from which features shall be selected (Pandas DataFrame)
    selection_method        method to use for selection, Mutual Information, F-Test, Boruta or ElasticNet
    target                  target variable (binary outcome)
    verbose                 tells us whether to print additional information (default=False)
    kwargs                  with kwargs one can provide specific instructions to any of the classifiers

    """
    def execute(
        self, data, selection_method=None, target=None, top=None, reverse=False, verbose=False, **kwargs
    ):
        try:
            if not data.empty:
                
                features, labels = data.drop(target, axis=1), data[target]

                if selection_method:
                    if verbose:
                        print(
                            f"Performing feature selection with {selection_method.__name__} ..."
                        )
                    
                    n_top = top or features.shape[1]
                    selector = selection_method(top=n_top, **kwargs)
                    
                    if verbose:
                        print(f"Total features prior to selection: {features.shape[1]}")
                        [
                            print(f"{feat}")
                            for feat in features.columns
                        ]
                    selector.fit(features.to_numpy(), labels.to_numpy())
                    
                    ''' Empty selector will return empty list '''
                    if len(selector.get_indices()) > 0:
                        features = features.iloc[:, selector.get_indices(reverse=reverse)]

                    if verbose:
                        print(f"Total features after selection: {features.shape[1]}")
                        [
                            print(f"{feat}")
                            for feat in features.columns
                        ]
                    data = pd.concat(
                        [features, labels.to_frame()], axis=1
                    )
                else:
                    print("No selection method provided, features remained untouched.")
            else:
                raise AttributeError("No data provided")

        except Exception:
            print(traceback.format_exc())
            logging.error(traceback.format_exc())

        return data, features.columns.tolist()
