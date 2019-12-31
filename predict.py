import os
from review_classification import Predict

path = os.path.abspath("models")
model_path = path + "/LogisticRegression.pkl"

lrm = Predict(model_path)

def predict(sentence):
    return lrm.predict(sentence)
