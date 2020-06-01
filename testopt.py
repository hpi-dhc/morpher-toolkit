import numpy as np

from morpher import config
from morpher.jobs import Evaluate, Impute, Load, Train
from morpher.metrics import get_discrimination_metrics

target = "AKI"

train = Impute().execute(
    Load().execute(source=config.FILE, filename="train"),
    imputation_method=config.DEFAULT,
)
test = Impute().execute(
    Load().execute(source=config.FILE, filename="test"),
    imputation_method=config.DEFAULT,
)

param_grid_lr = {
    "penalty": ["none", "l2"],
    "C": np.logspace(0, 4, 10),
    "solver": ["lbfgs"],
    "max_iter": [10000],
}

hyperparams_rf = {"n_estimators": 300, "max_depth": 2}

models = {}

models.update(
    Train().execute(
        train,
        target=target,
        optimize="yes",
        param_grid=param_grid_lr,
        algorithms=[config.LOGISTIC_REGRESSION],
    )
)
models.update(
    Train().execute(
        train,
        target=target,
        hyperparams=hyperparams_rf,
        algorithms=[config.RANDOM_FOREST],
    )
)

results = Evaluate().execute(test, target=target, models=models)

for algorithm in results:
    print("Metrics for {}".format(algorithm))
    print(get_discrimination_metrics(**results[algorithm]))
