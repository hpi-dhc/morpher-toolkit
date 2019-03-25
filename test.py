from morpher.load import *
from morpher.train import *
from morpher.split import *
from morpher.impute import *
from morpher.scale import *
from morpher.evaluate import *
from morpher import config
from pprint import pprint

data = load(file_name="full")

data = impute(data, method=config.DEFAULT)

train_data, test_data = split(data=data, test_size=0.3)

algorithms = train(train_data, target="AKI", algorithms=[config.DECISION_TREE, config.RANDOM_FOREST])

results = evaluate(test_data, target="AKI", algorithms=algorithms)

pprint(results)






















