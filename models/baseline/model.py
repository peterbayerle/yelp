from ..model import Model
from .pipeline import create_pipeline
from sklearn.metrics import classification_report

class Baseline(Model):
    def __init__(self, classifier):
        self.classifier = classifier
        self.train_time = 0

    def _train(self, x_train, y_train):
        self._trained_model = create_pipeline(self.classifier)
        self._trained_model.fit(x_train, y_train)

    def _evaluate(self, x_test, y_test):
        assert self.train_time

        y_pred = self._trained_model.predict(x_test)
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
        return report
