import pandas as pd
import numpy as np
import argparse
import joblib

from dask.distributed import Client, progress
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


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
        print("-" * 80)
        print(f"Predicting with a dataframe of {generator.n}")
        print("-" * 80)

        predictions = []

        predictions = self.model.predict(
            self.generator, workers=-1, verbose=True, steps=self.batch_size
        )
        self.predictions = np.argmax(predictions, axis=1)

    def classification_report(self):
        return classification_report(self.generator.labels, self.predictions)

    def confusion_matrix(self):
        return confusion_matrix(self.generator.labels, self.predictions)
