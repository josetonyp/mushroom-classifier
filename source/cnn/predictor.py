from __future__ import annotations
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


class Predictor(object):
    """
    A class for making predictions using a Keras model.

    Attributes:
        model (keras.models.Model): The Keras model to use for predictions.
        generator (object): The image
        data generator to use for predictions.
        batch_size (int): The batch size to use for predictions.
    """

    def __init__(self, batch_size: int = 64):
        """
        Initializes a new Predictor object.

        Args:
            batch_size (int): The batch size to use for predictions
            (default: 64).
        """
        self.__batch_size = batch_size

    def load(self, model_file: str) -> Predictor:
        """
        Loads a Keras model from a file.

        Args:
            model_file (str): The path to the model file.

        Returns:
            self (Predictor): The Predictor object.
        """
        self.model = load_model(model_file)

        return self

    def predict(self, generator: object) -> Predictor:
        """
        Makes predictions using the loaded Keras model and the given image
        data generator.

        Args:
            generator (object): The image data generator to use for
            predictions.

        Returns:
            predictions (numpy.ndarray): The predicted class probabilities
            for the input images.
        """
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
        return self

    def accuracy_score(self) -> float:
        """
        Computes the accuracy score of the predictions.

        Returns:
            score (float): The accuracy score.
        """
        return accuracy_score(self.generator.labels, self.predictions)

    def classification_report(self, to_file: str | None = None) -> str:
        """
        Generate a classification report for the model's predictions.

        Args:
            to_file (str | None): If provided, the classification report
            will be saved to this file.

        Returns:
            str: The classification report as a string.
        """
        self.class_report = classification_report(
            self.generator.labels, self.predictions
        )

        if to_file is not None and isinstance(to_file, str):
            open(to_file, "w", encoding="utf-8").write(self.class_report)

        return self.class_report

    def confusion_matrix(self, to_file: str | None = None) -> str:
        """
        Computes the confusion matrix for the predictions made by the model.

        Args:
            to_file (str | None): If provided, saves the confusion matrix
            as a JSON file at the given path.

        Returns:
            str: The computed confusion matrix.
        """
        self.cnf_matrix = confusion_matrix(
            self.generator.labels,
            self.predictions,
        )

        if to_file is not None and isinstance(to_file, str):
            text = json.dumps({"matrix": self.cnf_matrix.tolist()})
            open(to_file, "w", encoding="utf-8").write(text)

        return self.cnf_matrix
