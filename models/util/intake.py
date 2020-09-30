import time
import json
import numpy as np
from sklearn.model_selection import train_test_split

class ReviewReader(object):
    def __init__(self, path):
        self.path = path

    def _raw(self):
        with open(self.path) as f:
            for line in f:
                review_obj = json.loads(line)
                label = 1 if review_obj['stars'] >= 3.5 else 0
                yield review_obj['text'], label

    def split(self, test_size):
        x, y = list(), list()
        for text, label in self._raw():
            x.append(text)
            y.append(label)
        return train_test_split(x, y, test_size=test_size)
