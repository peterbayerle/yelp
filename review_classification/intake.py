import json
import linecache
from math import inf
from nltk import pos_tag, wordpunct_tokenize, sent_tokenize
import numpy as np
import pickle
from sklearn.model_selection import KFold

class YelpReviewReader(object):
    def __init__(self, path, max=10_000):
        self.path = path
        self.max = max

    def raw(self):
        with open(self.path) as f:
            count = 0
            for line in f:
                if count >= self.max:
                    break
                review_obj = json.loads(line)
                label = 1 if review_obj["stars"] >= 3.5 else 0
                yield (review_obj["text"], label)
                count += 1

    @classmethod
    def tokenize(self, sentence):
        words = wordpunct_tokenize(sentence)
        tokens = pos_tag(words)
        return tokens

    def tokens(self):
        for item in self.raw():
            review, label = item
            yield (self.tokenize(review), label)

    def save_tokens(self, path):
        document = list(self.tokens())
        pickle.dump(document, open(path, "wb"))

class YelpReviewLoader(object):
    def __init__(self, reviews, folds=12, shuffle=True):
        # reviews := [(tokens1, label1), (tokens2, label2), ...]
        # tokensX := [(wordX1, POSX1), (wordX2, POSX2), ...]
        self.folds = KFold(n_splits=folds, shuffle=True)
        self.shuffle = shuffle

        self.X = np.array([item[0] for item in reviews])
        self.y = np.array([item[1] for item in reviews])

    def __iter__(self):
        for train_index, test_index in self.folds.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            yield X_train, X_test, y_train, y_test
