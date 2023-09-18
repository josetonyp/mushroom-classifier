from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataSetGenerator:
    def __init__(self, preprocess_input_method):
        self.preprocess_input_method = preprocess_input_method

    def generator(
        self,
        data_source,
        dataset,
        images_folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
    ):
        match (data_source):
            case ("train"):
                return self.__train_generator().flow_from_dataframe(
                    dataset,
                    directory=images_folder,
                    x_col="feature",
                    y_col="label",
                    class_mode="sparse",
                    target_size=target_size,
                    batch_size=batch_size,
                    shuffle=True,
                )
            case ("valid"):
                return self.__test_generator().flow_from_dataframe(
                    dataset,
                    directory=images_folder,
                    x_col="feature",
                    y_col="label",
                    class_mode="sparse",
                    target_size=target_size,
                    batch_size=batch_size,
                    shuffle=True,
                )

    def __train_generator(self):
        return ImageDataGenerator(
            preprocessing_function=self.preprocess_input_method,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=1.1,
            horizontal_flip=True,
        )

    def __test_generator(self):
        return ImageDataGenerator(preprocessing_function=self.preprocess_input_method)
