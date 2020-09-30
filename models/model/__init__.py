import json
from ..util import time_this

class Model(object):
    def __init__(self):
        self.train_time = 0

    def _train(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def train(self, x_train, y_train, **kwargs):
        self._train = time_this(self._train)
        self.train_time = self._train(x_train, y_train, **kwargs)

    def evaluate(self, x_test, y_test, save_to=False):
        self._evaluate = time_this(self._evaluate)
        report, eval_time = self._evaluate(x_test, y_test)
        report['train_time'] = self.train_time
        report['eval_time'] = eval_time

        if save_to:
            with open(save_to, 'w') as f:
                json.dump(report, f, indent=4)

        return report
