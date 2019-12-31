from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .transform import Normalize

def create_pipeline(estimator, reduction=False):
    steps = [("normalize", Normalize()),
             ("tfidf", TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False))]

    if reduction == True:
        steps.append(("svd", TruncatedSVD()))

    steps.append(("classifier", estimator))
    return Pipeline(steps)

def identity(words):
    return words
