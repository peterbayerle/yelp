import os
from review_classification import Model
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

data_path = os.path.abspath("data")
# review_path = data_path + "/{}.json".format("review")
review_pkl_path = data_path + "/{}.pkl".format("review")

model_path = os.path.abspath("models")
model_pkl_path = model_path + "/LogisticRegression.pkl"

model = Model(review_path)

# model.pickle_data(25_000, review_pkl_path)
model.pkl_path = review_pkl_path

# model.evaluate([LogisticRegression, SGDClassifier, SVC], folds=12)
model.create(LogisticRegression, model_pkl_path)
