import json
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
import tensorflow as tf

tf.get_logger().setLevel("INFO")


class Predictor:
    def __init__(self, n_class=5, batch_size=64, target_file_size_shape=(224, 224)):
        self.batch_size = batch_size
        self.target_file_size = target_file_size_shape
        self.n_class = n_class
        print(f"Output classes {self.n_class}", end=",\n")

    def load(self, model_file):
        self.model = load_model(model_file)

        return self

    def predict(self, generator):
        self.generator = generator
        steps = (generator.n // self.batch_size) + 1

        print("-" * 80)
        print(f"Predicting with {generator.n} images")
        print(f"Predicting in {steps} steps")
        print("-" * 80)

        predictions = self.model.predict(self.generator, verbose=True, steps=steps)

        self.predictions = np.argmax(predictions, axis=1)

    def classification_report(self, to_file=None):
        report = classification_report(self.generator.labels, self.predictions)

        if to_file != None and type(to_file) == type(""):
            open(to_file, "w").write(report)

        return report

    def confusion_matrix(self, to_file=None):
        matrix = confusion_matrix(self.generator.labels, self.predictions)

        if to_file != None and type(to_file) == type(""):
            open(to_file, "w").write(json.dumps({"matrix": matrix.tolist()}))

        return matrix
