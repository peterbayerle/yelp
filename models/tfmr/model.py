from .helpers import *
from ..model import Model
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras import activations, optimizers, losses
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

class BERT(Model):
    def __init__(self, max_len):
        self.model_name = 'distilbert-base-uncased'
        self.max_len = max_len
        self.tkzr = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name)
        self.optimizer = optimizers.Adam(learning_rate=3e-5)
        self.loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    def _train(self, x_train, y_train, batch_size=1, n_epochs=1):
        feature_list = construct_feature_list(x_train, self.tkzr, max_len=self.max_len)
        tfdataset = to_tfdataset(feature_list, y_train)
        tfdataset = tfdataset.batch(batch_size)
        train_size = len(x_train)

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        self.model.fit(
            tfdataset,
            batch_size=batch_size,
            steps_per_epoch = int(train_size / batch_size),
            epochs=n_epochs
        )

    def _evaluate(self, x_test, y_test):
        feature_list = construct_feature_list(x_test, self.tkzr, max_len=self.max_len)
        tfdataset = to_tfdataset(feature_list)
        tfdataset = tfdataset.batch(1)

        preds = self.model.predict(tfdataset, verbose=1, steps=len(x_test))[0]
        preds = activations.softmax(tf.convert_to_tensor(preds)).numpy()
        y_pred = np.argmax(preds, axis=1)

        return classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
