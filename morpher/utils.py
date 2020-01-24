import jsonpickle as jp
import inspect
import jsonpickle.ext.numpy as jsonpickle_numpy
import time
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer

jsonpickle_numpy.register_handlers()


def pickle(obj, path=None):

    try:
        frozen = jp.encode(obj)
        if not path:
            path = retrieve_name(obj)
        with open(path, 'w') as file:
            print("Pickling object...")
            file.write(frozen)
        return True
    except Exception as e:
        print(f"Could not pickle object. {e}")
        return False


def unpickle(path):
    try:
        with open(path, 'r') as file:
            frozen = file.read()

        print("Unpickling object...")
        return jp.decode(frozen)

    except Exception as e:
        print(f"Could not unpickle object. {e}")
        return False


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def data_description_to_csv(data, target, path, print_summary=False):

    if data is None:
        raise AttributeError("No data has been loaded.")

    summary = (
        "### Data description:\n"
        "# Rows: {0}\n"
        "# Columns: {1}\n"
        "# Column names: {2}"
    ).format(data.shape[0], data.shape[1], data.columns.values)

    summary += ("\n### Descriptive Statistics for Features Used\n")
    bool_cols = [col for col in data if data[col].dropna().value_counts().index.isin([0, 1]).all()]
    data_description = data.describe(include='all')

    pvalues = p_values(data, target)
    pvalues[target] = .0  # p-value for the target is by definition 0

    # initialize frame header
    header = ["Column", "All(n={0})".format(data.shape[0]), "Missing", "Missing(%)", "P"]
    stats = pd.DataFrame(columns=header)

    for column in data.columns:
        if column in bool_cols:
            count = data.loc[data[column] == 1].shape[0]
            stats.loc[len(stats)] = [column, "{0} ({1}%)".format(count, round((count / data[column].dropna().count()) * 100, 2)), data[column].isnull().sum(), round((data[column].isnull().sum() / data.shape[0]) * 100, 1), round(pvalues[column], 3)]
        else:
            stats.loc[len(stats)] = [column, "{0}+/-{1}".format(round(data_description[column]["mean"], 2), round(data_description[column]["std"], 2)), data[column].isnull().sum(), round((data[column].isnull().sum() / data.shape[0]) * 100, 1), round(pvalues[column], 3)]

    stats.to_csv(path, index=False, sep=";")
    print(f"Saved description to {path}.")

    if print_summary:
        print(summary)

    return stats


def p_values(data, target, print_summary=False):
    '''
    Prints statistical information such as p-values
    '''
    if data is None:
        raise AttributeError("No data available, please load some data.")

    '''
    Initializes pipeline by imputing and normalizing data
    '''
    labels = data[target]
    features = data.drop(target, axis=1)
    columns = features.columns.values
    X = pd.DataFrame(Normalizer().fit_transform(SimpleImputer(strategy='mean').fit_transform(features)), columns=columns)
    y = labels
    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()
    if print_summary:
        print(est.summary())

    return est.pvalues


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
