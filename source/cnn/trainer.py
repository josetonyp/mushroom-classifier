from __future__ import annotations
from math import ceil
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .callbacks.timing import TimingCallback

tf.get_logger().setLevel("INFO")


class Trainer(object):
    """
    A class for training a Keras model using a generator for image data.

    Args:
        model (keras.Model): The Keras model to train.
        n_class (int): The number of output classes.
        batch_size (int): The batch size for training.
        compile_metrics (list of str): The metrics to use for model
        compilation.
        target_file_size_shape (tuple of int): The target size for input
        images.
        logger (object or None): A logger object for logging training
        information.
        label_feature (str): The name of the label feature in the generator.
        image_feature (str): The name of the image feature in the generator.
        epochs (int): The number of epochs to train for.

    Methods:
        train(train_generator, valid_generator): Trains the model using the
        given generators.
        save(file): Saves the trained model to a file.
        save_history(file): Saves the training history to a file.

    """

    def __init__(
        self,
        model: Model,
        n_class: int = 5,
        batch_size: int = 64,
        compile_metrics: list(str) = ["accuracy"],
        target_file_size_shape: tuple = (224, 224),
        logger: object | None = None,
        label_feature: str = "label",
        image_feature: str = "image_lien",
        epochs: int = 8,
    ) -> None:
        """
        Initializes a new Trainer object.

        Args:
            model (keras.Model): The Keras model to train.
            n_class (int): The number of output classes.
            batch_size (int): The batch size for training.
            compile_metrics (list of str): The metrics to use for model
            compilation.
            target_file_size_shape (tuple of int): The target size for input
            images.
            logger (object or None): A logger object for logging training
            information.
            label_feature (str): The name of the label feature in the
            generator.
            image_feature (str): The name of the image feature in the
            generator.
            epochs (int): The number of epochs to train for.
        """
        self.model = model
        self.batch_size = batch_size
        self.target_file_size = target_file_size_shape
        self.label_feature = label_feature
        self.image_feature = image_feature
        self.n_class = n_class
        self.compile_optimizer = "adam"
        self.compile_metrics = compile_metrics
        self.preprocess_input_method = None
        self.logger = logger
        self.epochs = epochs

    def train(
        self,
        train_generator: object,
        valid_generator: object,
    ) -> Trainer:
        """
        Trains the model using the given generators.

        Args:
            train_generator (object): The generator for training data.
            valid_generator (object): The generator for validation data.

        Returns:
            The Trainer object.
        """
        self.logger.info("-" * 80)
        self.logger.info("Training Characterization")
        self.logger.info(f"Output classes {self.n_class}")
        self.logger.info(f"Compiling with optimizer {self.compile_optimizer}")
        self.logger.info(f"Compiling in {self.batch_size} batch size")
        self.logger.info(
            (
                f"Compiling for {train_generator.n//self.batch_size + 1}"
                "steps_per_epoch"
            )
        )
        self.logger.info(f"Training with {train_generator.n} of images")
        self.logger.info(f"Generator has {len(train_generator)} batches")
        self.logger.info("-" * 80)

        history = self.model.fit(
            train_generator,
            steps_per_epoch=ceil(train_generator.n / self.batch_size),
            epochs=self.epochs,
            workers=-1,
            validation_data=valid_generator,
            validation_steps=ceil(valid_generator.n / self.batch_size),
            callbacks=self.__callbacks(),
        )

        self.logger.info(history)
        self.history = history

        return self

    def save(self, file: str) -> Trainer:
        """
        Saves the trained model to a file.

        Args:
            file (str): The path to the file to save the model to.

        Returns:
            The Trainer object.
        """
        self.logger.info("-" * 80)
        self.logger.info("Saving Results")
        self.logger.info(f"Saving model to file {file}")
        self.model.save(file)
        return self

    def save_history(self, file: str) -> Trainer:
        """
        Saves the training history to a file.

        Args:
            file (str): The path to the file to save the history to.

        Returns:
            The Trainer object.
        """
        pd.DataFrame(self.history.history).to_csv(file)
        return self

    def __callbacks(self) -> list:
        """
        Returns a list of Keras callbacks to use during training.

        Returns:
            A list of Keras callbacks.
        """
        early_stopping = EarlyStopping(
            monitor="loss",
            min_delta=1e-3,
            patience=5,
            verbose=1,
            mode="min",
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
