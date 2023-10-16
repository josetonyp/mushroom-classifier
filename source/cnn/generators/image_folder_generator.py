from struct import unpack
import tensorflow
from tensorflow.image import decode_jpeg, resize
from tensorflow.io import read_file
from tqdm import tqdm
import numpy as np
import pandas as pd


class JPEGInspector:
    """Inspect JPEG image data and validates its internal
    structure.
    """

    MARKERS = {
        0xFFD8: "Start of Image",
        0xFFE0: "Application Default Header",
        0xFFDB: "Quantization Table",
        0xFFC0: "Start of Frame",
        0xFFC4: "Define Huffman Table",
        0xFFDA: "Start of Scan",
        0xFFD9: "End of Image",
    }

    def __init__(self, image_file: str) -> None:
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def decode(self) -> None:
        """
        Decodes the image data in the `img_data` attribute using
        the JPEG decoding algorithm.

        Returns:
            None
        """
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return
            elif marker == 0xFFDA:
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                lenchunk = 2 + lenchunk
                data = data[lenchunk:]
            if len(data) == 0:
                break


class DataSetFolderGenerator:
    """Generates a TensorFlow dataset from a folder containing"""

    def __init__(self, preprocess_input_method=None):
        self.preprocess_input_method = preprocess_input_method

    def generator(
        self,
        df: pd.DataFrame,
        target_size: tuple = (150, 150),
        batch_size: int = 32,
        feature_name: str = "filepath",
    ) -> tensorflow.Tensor:
        """
        Generates a TensorFlow dataset from a Pandas DataFrame containing
        image filepaths and labels.

        Args:
            df (object): The Pandas DataFrame containing the image filepaths
            and labels.
            target_size (tuple, optional): The target size of the images.
            Defaults to (150, 150).
            batch_size (int, optional): The batch size. Defaults to 32.
            feature_name (str, optional): The name of the column containing
            the image filepaths. Defaults to "filepath".

        Returns:
            tensorflow.Tensor: The TensorFlow dataset.
        """
        bads = self.__find_bad_images(df[feature_name])
        df = self.__remove_broken_images(bads, df, feature_name)

        dataset_train = tensorflow.data.Dataset.from_tensor_slices(
            (df[feature_name], df.label)
        )

        dataset_train = dataset_train.map(
            lambda x, y: [
                self.load_and_resize_image(
                    x,
                    filse_size=target_size,
                ),
                y,
            ],
            num_parallel_calls=-1,
        ).batch(batch_size)

        dataset_train.n = df.shape[0]
        dataset_train.labels = df.label

        return dataset_train

    @tensorflow.function
    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load an image from a file path and return it as a numpy array.

        Args:
            filepath (str): The path to the image file.

        Returns:
            np.ndarray: The loaded image as a numpy array.
        """
        im = read_file(filepath)
        return decode_jpeg(im, channels=3)

    @tensorflow.function
    def load_and_resize_image(
        self,
        filepath: str,
        filse_size: tuple = (256, 256),
    ) -> np.ndarray:
        """
        Load an image from the given filepath and resize it to the
        specified size.

        Args:
            filepath (str): The path to the image file.
            file_size (tuple, optional): The desired size of the image.
            Defaults to (256, 256).

        Returns:
            np.ndarray: The resized image as a NumPy array.
        """
        im = self.load_image(filepath)
        return resize(im, filse_size)

    def __find_bad_images(self, df: pd.DataFrame) -> list:
        """
        Finds and returns a list of filepaths for images that are broken
        or do not exist.

        Args:
            df (object): A pandas DataFrame containing the filepaths
            of the images.

        Returns:
            A list of filepaths for images that are broken or do not exist.
        """
        bads = []
        print("Searching for broken or unexisting JPG Images")
        for filepath in tqdm(df):
            try:
                image = JPEGInspector(filepath)
                image.decode()
            except KeyError:
                bads.append(filepath)

        return bads

    def __remove_broken_images(
        self,
        bads: list,
        df: pd.DataFrame,
        feature_name: str,
    ) -> pd.DataFrame:
        """
        Removes broken images from the dataset.

        Args:
            bads (list): A list of broken image filenames.
            df (pd.DataFrame): The dataset containing the image filenames.
            feature_name (str): The name of the column containing the
            image filenames.

        Returns:
            pd.DataFrame: The updated dataset with the broken images removed.
        """
        if bads != []:
            print("Removing the following images from the dataset")
            for bad in bads:
                print(f"- {bad}")
                df.drop(df.loc[df[feature_name] == bad].index, inplace=True)

        return df
