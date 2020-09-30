from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .transform import Normalize, Tokenize

def create_pipeline(classifier):
    steps = [
        ('tokenize', Tokenize()),
        ('normalize', Normalize()),
        ('tfidf', TfidfVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False)),
        ('classifier', classifier())
    ]

    return Pipeline(steps)
