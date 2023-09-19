import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

tf.get_logger().setLevel("INFO")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .training_history_image import TrainingHistoryImage
from .callbacks.timing import TimingCallback


class Trainer(object):
    def __init__(
        self,
        model,
        n_class=5,
        batch_size=64,
        compile_metrics=["accuracy"],
        target_file_size_shape=(224, 224),
        logger=None,
        label_feature="label",
        image_feature="image_lien",
    ):
        ### Variables
        self.model = model
        self.batch_size = batch_size

        ### Constants
        self.target_file_size = target_file_size_shape
        self.label_feature = label_feature
        self.image_feature = image_feature
        self.n_class = n_class

        self.compile_optimizer = "adam"
        self.compile_metrics = compile_metrics

        self.preprocess_input_method = None

        if logger == None:

            class PLogger:
                def info(self, text):
                    self.logger.info(text)

                def debug(self, text):
                    self.logger.info(text)

            self.logger = PLogger()
        else:
            self.logger = logger

    def train(self, train_generator, valid_generator):
        self.logger.info("-" * 80)
        self.logger.info("Training Characterization")
        self.logger.info(f"Output classes {self.n_class}")
        self.logger.info(f"Compiling with optimizer {self.compile_optimizer}")
        self.logger.info(f"Compiling in {self.batch_size} batch size")
        self.logger.info(
            f"Compiling for {train_generator.n//self.batch_size + 1} epochs"
        )
        self.logger.info(f"Training with a dataframe of {train_generator.n}")
        self.logger.info("-" * 80)

        history = None
        train_batches = train_generator.n // self.batch_size + 1

        history = self.model.fit(
            train_generator,
            steps_per_epoch=self.batch_size,
            epochs=train_batches,
            workers=-1,
            validation_data=valid_generator,
            validation_steps=self.batch_size,
            callbacks=self.__callbacks(),
        )
        self.logger.info(history)
        self.history = history

        return self

    def save(self, file):
        self.logger.info("-" * 80)
        self.logger.info("Saving Results")
        self.logger.info(f"Saving model to file {file}/model.keras")
        self.model.save(f"{file}/model.keras")

        try:
            self.logger.info(
                f"Saving training history to file {file}/train_history.json"
            )
            with open(f"{file}/train_history.json", "w") as outfile:
                json.dump(self.history.history, outfile)

            self.logger.info(
                f"Saving training history to file {file}/train_history.csv"
            )
            pd.DataFrame(self.history.history).to_csv(f"{file}/train_history.csv")

            self.logger.info(f"Creating training image {file}/train_history.jpg")
            TrainingHistoryImage(f"{file}/train_history.csv").render().save(
                f"{file}/train_history.jpg"
            )
        except Exception as e:
            self.logger.info(e)
            pass
        self.logger.info("-" * 80)

    def __callbacks(self):
        early_stopping = EarlyStopping(
            monitor="loss", min_delta=1e-3, patience=5, verbose=1, mode="min"
        )

        reduce_learning_rate = ReduceLROnPlateau(
            monitor="loss",
            min_delta=1e-3,  # episilon= 0.001,
            patience=3,
            factor=0.1,
            cooldown=4,
            verbose=1,
        )

        return [early_stopping, reduce_learning_rate, TimingCallback()]
