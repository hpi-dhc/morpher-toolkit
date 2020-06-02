# MORPHER Toolkit

This library provides ready-to-use capabilities to automate common tasks necessary for clinical predictive modeling. It provides a common interface to enable extension of existing machine learning algorithms.

## Model Protyping

Among other functions, it provides encapsulated functionalities for the following routine tasks in clinical modeling:

*   Imputation
*   Transformation
*   Training
*   Evaluation and
*   Interpretation

## Developing and Evaluating a Model

To start, clone the repository and run `pip install -e .`. This command will install MORPHER Toolkit and its dependencies to your current environment. We recommended setting up a dedicated environment using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html "Conda") or your favorite environment manager. The option `-e` makes sure that any local changes you make to the library, e.g., adding now jobs, are directly visible enviroment-wide.
Some dependencies may require compilation. For this, install `gcc` and `g++` with your distribution's package manager.

The toolkit functionalities are exposed by means of MORPHER Jobs to impute, train, and evaluate models. Import the specific modules and classes in your code:

```python
import morpher
from morpher.config import algorithms, imputers
from morpher.jobs import *
from morpher.metrics import *
```

Say you are interested to test different models on on the task of predicting Acute Kidney Injury (AKI). Assuming you have a extracted a .CSV file named `cohort.csv` with a number of numeric features for the target `AKI` , you can generate a Receiver Operating Characteristc Curve (ROC) curve with the following commands:

```python

target = "AKI"

''' First we load, impute and split the dataset in train and test '''
data = Load().execute(source=config.FILE, filename="cohort.csv")
data = Impute().execute(data, imputation_method=imputers.DEFAULT)
train, test = Split().execute(data, test_size=0.2)

''' The we train the given algorithms on the training set '''
models = Train().execute(
    train, 
    target=target,
    algorithms=[algorithms.LR, algorithms.DT, algorithms.RF, algorithms.GBDT, algorithms.MLP]
)

''' And evaluate them on the test set '''
results = Evaluate().execute(test, target=target, models=models)
```

In the code above, any of the config parameters can be changed to the existing MORPHER Toolkit options. The dictionary `results` has the results of applying the trained models on the test set.

To plot the ROC curve, we need to import the plotting modules:

```python
from morpher.plots import *
import matplotlib.pyplot as plt
plot_roc(results)
plt.show()
```

<img src="https://i.ibb.co/M9TpM5F/roc.png"
     alt="ROC Curve"
     style="float: left; margin-right: 10px;" width="512" />

To explain the model you just created, we need to run the Explain job using one of the available explainers, such as feature contribution or LIME. The dictionary `explanations` contains the feature rankings indexed by each algorithm.

```python

from morpher.config import explainers
explanations = Explain().execute(
    train,
    models=models,
    explainers = [explainers.FEAT_CONTRIB, explainers.LIME],
    target=target,
    exp_args = {'test': test}                 
)

plot_explanation_heatmap(explanations[algorithms.RF])

```
You can check another examples using Jupyter under `notebooks`, such as for predicting diabetes.

## Library Dependencies

*   Python 3.6+
*   scikit-learn
*   lime
*   scipy
*   matplotlib
*   sklearn_pandas
*   fancyimpute
*   imbalanced-learn
*   statsmodels
*   jsonpickle
*   jinja2
*   mpld3
*   shap
*   simplejson
*   lightgbm
*   xgboost
