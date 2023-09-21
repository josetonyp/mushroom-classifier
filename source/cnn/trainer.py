import pandas as pd
import numpy as np
import json, math

import tensorflow as tf

tf.get_logger().setLevel("INFO")
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
        self.logger = logger

    def train(self, train_generator, valid_generator):
        self.logger.info("-" * 80)
        self.logger.info("Training Characterization")
        self.logger.info(f"Output classes {self.n_class}")
        self.logger.info(f"Compiling with optimizer {self.compile_optimizer}")
        self.logger.info(f"Compiling in {self.batch_size} batch size")
        self.logger.info(
            f"Compiling for {train_generator.n//self.batch_size + 1} steps_per_epoch"
        )
        self.logger.info(f"Training with {train_generator.n} of images")
        self.logger.info(f"Generator has {len(train_generator)} batches")
        self.logger.info("-" * 80)

        history = None
        # The batch size determines how many of the images are shown per one step
        # steps_per_epoch: Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
        history = self.model.fit(
            train_generator,
            steps_per_epoch=math.ceil(train_generator.n / self.batch_size),
            epochs=8,
            workers=-1,
            validation_data=valid_generator,
            validation_steps=math.ceil(valid_generator.n / self.batch_size),
            callbacks=self.__callbacks(),
        )
        self.logger.info(history)
        self.history = history

        return self

    def history(self):
        self.history

    def save(self, file):
        self.logger.info("-" * 80)
        self.logger.info("Saving Results")
        self.logger.info(f"Saving model to file file")
        self.model.save(file)

        # try:
        #     self.logger.info(
        #         f"Saving training history to file {folder}/train_history.csv"
        #     )
        #     pd.DataFrame(self.history.history).to_csv(f"{folder}/train_history.csv")

        #     self.logger.info(f"Creating training image {folder}/train_history.jpg")
        #     TrainingHistoryImage(f"{folder}/train_history.csv").render().save(
        #         f"{folder}/train_history.jpg"
        #     )
        # except Exception as e:
        #     self.logger.info(e)
        #     pass
        # self.logger.info("-" * 80)

    def history(self, file):
        try:
            self.logger.info(f"Saving training history to file {file}")
            with open(file, "w") as outfile:
                json.dump(self.history.history, outfile)
        except Exception as e:
            self.logger.info(e)
            self.logger.info("Histtory:")
            self.logger.info(json.dumps(self.history.history))
            pass

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
