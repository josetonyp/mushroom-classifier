from tensorflow.keras.preprocessing.image import ImageDataGenerator


class FolderGenerator:
    def __init__(self, preprocess_input_method):
        self.preprocess_input_method = preprocess_input_method

    def generator(
        self,
        data_source,
        folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode="sparse",
    ):
        match (data_source):
            case ("train"):
                return self.__train_generator().flow_from_directory(
                    folder,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=class_mode,
                    interpolation="bicubic",
                    shuffle=True,
                )
            case ("valid"):
                return self.__test_generator().flow_from_directory(
                    folder,
                    target_size=target_size,
                    batch_size=batch_size,
                    class_mode=class_mode,
                    interpolation="bicubic",
                    shuffle=True,
                )

    def __train_generator(self):
        return ImageDataGenerator(
            preprocessing_function=self.preprocess_input_method,
            rotation_range=5,
            zoom_range=[0.95, 1.05],
            horizontal_flip=True,
            dtype=int,
        )

    def __test_generator(self):
        return ImageDataGenerator(
            preprocessing_function=self.preprocess_input_method,
            dtype=int,
        )
