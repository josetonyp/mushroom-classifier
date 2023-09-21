from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import pandas as pd
from glob import glob
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize


class ImageFolderGenerator:
    def __init__(self, preprocess_input_method=None):
        self.preprocess_input_method = preprocess_input_method

    def generator(
        self,
        data_source,
        folder,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
    ):
        liste = glob(f"{folder}/**/*")
        liste = list(map(lambda x: [x, x.split("/")[-2]], liste))
        df = pd.DataFrame(liste, columns=["filepath", "nameLabel"])
        df["label"] = df["nameLabel"].replace(
            df.nameLabel.unique(), [*range(len(df.nameLabel.unique()))]
        )
        dataset_train = tensorflow.data.Dataset.from_tensor_slices(
            (df.filepath, df.label)
        )

        dataset_train = dataset_train.map(
            lambda x, y: [self.load_and_resize_image(x, filse_size=target_size), y],
            num_parallel_calls=-1,
        ).batch(batch_size)

        dataset_train.n = df.shape[0]
        dataset_train.labels = df.label

        return dataset_train

    @tensorflow.function
    def load_image(self, filepath):
        im = read_file(filepath)
        return decode_jpeg(im, channels=3)

    @tensorflow.function
    def load_and_resize_image(self, filepath, filse_size=(256, 256)):
        im = self.load_image(filepath)
        return resize(im, filse_size)
