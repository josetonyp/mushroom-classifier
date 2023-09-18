from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential


class AchB:
    def __init__(
        self, base_model, output_n_class, optimizer="adam", compile_metrics=["accuracy"]
    ):
        self.base_model = base_model
        self.compile_optimizer = optimizer
        self.compile_metrics = compile_metrics
        self.n_class = output_n_class

    def build(self, n_workers=-1):
        self.model = Sequential()
        self.model.add(self.base_model.model())
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation="relu"))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(self.n_class, activation="softmax"))

        self.model.compile(
            optimizer=self.compile_optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=self.compile_metrics,
        )

        return self.model

    def summary(self):
        return self.model.summary()
