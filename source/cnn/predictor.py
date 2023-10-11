import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("INFO")


class Predictor:
    def __init__(self, batch_size=64):
        self.__batch_size = batch_size

    def load(self, model_file):
        self.model = load_model(model_file)

        return self

    def predict(self, generator):
        self.generator = generator
        steps = (generator.n // self.__batch_size) + 1

        print("-" * 80)
        print(f"Predicting with {generator.n} images")
        print(f"Predicting in {steps} steps")
        print("-" * 80)

        predictions = self.model.predict(
            self.generator,
            verbose=True,
            steps=steps,
        )

        self.predictions = np.argmax(predictions, axis=1)

    def accuracy_score(self):
        return accuracy_score(self.generator.labels, self.predictions)

    def classification_report(self, to_file=None):
        self.class_report = classification_report(
            self.generator.labels, self.predictions
        )

        if to_file is None and type(to_file) is str:
            open(to_file, "w").write(self.class_report)

        return self.class_report

    def confusion_matrix(self, to_file=None):
        self.cnf_matrix = confusion_matrix(
            self.generator.labels,
            self.predictions,
        )

        if to_file is not None and type(to_file) is str:
            text = json.dumps({"matrix": self.cnf_matrix.tolist()})
            open(to_file, "w").write(text)

        return self.cnf_matrix
