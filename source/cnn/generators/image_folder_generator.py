from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
import pandas as pd
from glob import glob
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize

from struct import unpack
from tqdm import tqdm
import os


class JPEGInspector:
    MARKERS = {
        0xFFD8: "Start of Image",
        0xFFE0: "Application Default Header",
        0xFFDB: "Quantization Table",
        0xFFC0: "Start of Frame",
        0xFFC4: "Define Huffman Table",
        0xFFDA: "Start of Scan",
        0xFFD9: "End of Image",
    }

    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return
            elif marker == 0xFFDA:
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                data = data[2 + lenchunk :]
            if len(data) == 0:
                break


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

        ## Remove images that have internal errors
        bads = []
        print("Searching for broken JPG Images")
        for filepath in tqdm(df.filepath):
            image = JPEGInspector(filepath)
            try:
                image.decode()
            except:
                bads.append(filepath)

        if bads != []:
            print("Removing the following images from the dataset")
            for bad in bads:
                print(f"- {bad}")
                df.drop(df.loc[df["filepath"] == bad].index, inplace=True)

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


class DataSetFolderGenerator:
    def __init__(self, preprocess_input_method=None):
        self.preprocess_input_method = preprocess_input_method

    def generator(
        self,
        data_source,
        df,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        feature_name="filepath",
        label_name="label",
    ):
        ## Remove images that have internal errors

        bads = []
        print("Searching for broken or unexisting JPG Images")
        for filepath in tqdm(df[feature_name]):
            try:
                image = JPEGInspector(filepath)
                image.decode()
            except:
                bads.append(filepath)

        if bads != []:
            print("Removing the following images from the dataset")
            for bad in bads:
                print(f"- {bad}")
                df.drop(df.loc[df[feature_name] == bad].index, inplace=True)

        dataset_train = tensorflow.data.Dataset.from_tensor_slices(
            (df[feature_name], df.label)
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
