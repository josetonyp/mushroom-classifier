from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


class AchA:
    def __init__(
        self,
        base_model: object,
        output_n_class: int,
        optimizer: str = "adam",
        compile_metrics: list = ["accuracy"],
    ):
        self.__base_model = base_model
        self.__compile_optimizer = optimizer
        self.__compile_metrics = compile_metrics
        self.__n_class = output_n_class

    def build(self):
        model = Sequential()
        model.add(self.__base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(rate=0.3))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.__n_class, activation="softmax"))

        model.compile(
            optimizer=self.__compile_optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=self.__compile_metrics,
        )

        return model
