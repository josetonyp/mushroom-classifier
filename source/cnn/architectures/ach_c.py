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
        base_model,
        output_n_class,
        optimizer="adam",
        compile_metrics=["accuracy"],
        file_size=(254, 254),
    ):
        self.base_model = None
        self.compile_optimizer = optimizer
        self.compile_metrics = compile_metrics
        self.n_class = output_n_class
        self.file_size = file_size

    def build(self):
        self.model = Sequential(
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
                Dense(self.n_class, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer=self.compile_optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=False),
            metrics=self.compile_metrics,
        )

        return self.model

    def summary(self):
        return self.model.summary()
