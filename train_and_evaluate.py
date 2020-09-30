import json
from models.baseline import Baseline
from models.tfmr import BERT
from models.util import ReviewReader
import os
import pprint
from sklearn.linear_model import LogisticRegression

### reading/splitting data
data_path = os.path.join('data', 'yelp_review_20k.json')
rr = ReviewReader(data_path)
x_train, x_test, y_train, y_test = rr.split(1/2)

### get baseline model and transformer performance on held out test set
models = [
    ('lr', Baseline(LogisticRegression), {}),
    ('bert', BERT(max_len=500), {'batch_size': 16, 'n_epochs': 1})
]

for model_name, model, train_params in models:
    model.train(x_train, y_train, **train_params)
    report_path = os.path.join('performance', f'{model_name}_report.json')
    report = model.evaluate(x_test, y_test, save_to=report_path)
    pprint.pprint(report)
