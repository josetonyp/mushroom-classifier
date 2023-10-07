from glob import glob
import pandas as pd

from .base import Base


class FolderDataset(Base):
    """Loads images from a folder where images are classified by subfolders as classes"""

    def __init__(
        self,
        folder: str,
    ):
        self.__folder = folder

    def load(self):
        """Loads and extracts the image files and labels"""
        files = glob(f"{self.__folder}/**/*")
        labels = list(map(lambda x: x.split("/")[-2], files))

        self.__df = self.df = pd.DataFrame(
            {"feature": files, "label": labels, "label_name": labels}
        )
        self.label_names = self.df.label_name.value_counts().index

        factorization = {
            v: i for i, v in enumerate(self.df.label_name.value_counts().index)
        }
        self.df["label"] = self.df.label.replace(factorization)

        return self
