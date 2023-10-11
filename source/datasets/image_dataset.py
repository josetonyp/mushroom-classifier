import pandas as pd
import os, shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from .base import Base


class ImageDataSet(Base):
    """Loads a DataSet in order to prepare it for training or prediction"""

    def __init__(
        self,
        filename: str,
        image_feature: str,
        label_feature: str,
        image_folder: str = "",
    ):
        self.__filename = filename
        self.__image_feature = image_feature
        self.__label_feature = label_feature
        self.__image_folder = image_folder

    def load(self):
        """Loads and extracts the image files and labels"""
        self.__df = pd.read_csv(self.__filename, low_memory=False)
        self.label_statistics = self.__df.label.value_counts()
        self.n_class = len(self.label_statistics.values)
        self.df = pd.DataFrame(
            {
                "feature": self.__df[self.__image_feature],
                "label": self.__df[self.__label_feature],
                "label_name": self.__df[self.__label_feature],
            }
        )

        if self.__image_folder != "":
            self.df["feature"] = self.df.feature.apply(
                lambda x: f"{self.__image_folder}/{x}"
            )

        return self
