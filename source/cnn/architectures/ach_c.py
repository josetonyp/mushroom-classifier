from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential


class AchC:
    def __init__(
        self,
        base_model: object,
        output_n_class: int,
        optimizer: str = "adam",
        compile_metrics: list = ["accuracy"],
    ):
        self.__compile_optimizer = optimizer
        self.__compile_metrics = compile_metrics
        self.__n_class = output_n_class

    def build(self):
        model = Sequential(
            [
                Rescaling(
                    1.0 / 255,
                    input_shape=(
                        self.file_size[0],
                        self.file_size[1],
                        3,
                    ),
                ),
                Conv2D(16, 3, padding="same", activation="relu"),
                MaxPooling2D(),
                BatchNormalization(),
                Conv2D(32, 3, padding="same", activation="relu"),
                MaxPooling2D(),
                BatchNormalization(),
                Conv2D(64, 3, padding="same", activation="relu"),
                MaxPooling2D(),
                Flatten(),
                Dense(128, activation="relu"),
                Dense(self.__n_class, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=self.__compile_optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=self.__compile_metrics,
        )

        return model
