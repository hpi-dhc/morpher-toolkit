from collections import namedtuple

import pandas as pd
import numpy as np
from fancyimpute import KNN, SoftImpute
from sklearn.impute import SimpleImputer


class KNNImputer:
    """
    Perform KNN imputation.
    """

    def __init__(self, k=10):
        """
        Params:

        k                number of nearest neighbors to consider

        """

        self._imputer = KNNExt(k=k)

    def fit_transform(self, df):
        """
        run feature imputation

        Params:

        df               input data to run the imputation on
        """

        return pd.DataFrame(
            data=self._imputer.fit_transform(df.to_numpy()),
            columns=df.columns,
            index=df.index,
        )

    def fit(self, df):

        print("*** Fitting kNN imputer...")
        self._imputer.fit(df.to_numpy())

    def transform(self, df):
        print("*** Performing imputation using fitted kNN imputer...")
        return pd.DataFrame(
            data=self._imputer.transform(df.to_numpy()),
            columns=df.columns,
            index=df.index,
        )


class SoftImputer:
    """
    Perform matrix completion by iterative soft thresholding of SVD decompositions.
    Note that it does not support the methods 'fit' and 'transform'
    For compatibility, we included the two methods, but they refit the imputer, cf. KNNExt
    For more information, please check inductive vs. transductive imputation: https://github.com/iskandr/fancyimpute
    """

    def __init__(self):
        """
        Params:

        k                number of nearest neighbors to consider

        """

        self._imputer = SoftImpute()

    def fit_transform(self, df):
        """
        run feature imputation

        Params:

        df               input data to run the imputation on
        """

        return pd.DataFrame(
            data=self._imputer.fit_transform(df.to_numpy()),
            columns=df.columns,
            index=df.index,
        )

    def fit(self, df):
        return

    def transform(self, df):
        return self.fit_transform(df)


class KNNExt(KNN):
    """
    Extends the standard implementation of KNN imputation to allow for fit / transform to take place.
    For more information, please check inductive vs. transductive imputation: https://github.com/iskandr/fancyimpute
    TODO: find a better solution for this, for example, extending the KNN class in a separate file
    """

    def fit(self, X, y=None):
        if self.normalizer is not None:
            self.normalizer.fit(X)

    def transform(self, X, y=None):

        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X = X_original.copy()
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        X_filled = self.fill(X, missing_mask, inplace=True)
        if not isinstance(X_filled, np.ndarray):
            raise TypeError(
                "Expected %s.fill() to return NumPy array but got %s"
                % (self.__class__.__name__, type(X_filled))
            )

        X_result = self.solve(X_filled, missing_mask)
        if not isinstance(X_result, np.ndarray):
            raise TypeError(
                "Expected %s.solve() to return NumPy array but got %s"
                % (self.__class__.__name__, type(X_result))
            )

        X_result = self.project_result(X=X_result)
        X_result[observed_mask] = X_original[observed_mask]
        return X_result


_options = {
    "DEFAULT": SimpleImputer,
    "KNN": KNNImputer,
    "KNNext": KNNExt,
    "SOFT": SoftImputer,
}

imputer_config = namedtuple("options", _options.keys())(**_options)
