import pandas as pd
from shutil import copyfile
from os import mkdir
from os.path import exists
from tqdm import tqdm


class FolderBuilder:
    """Builds a folder from a Pandas Dataframe with one subfolder per label"""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe

    def build_folder(
        self,
        images_folder: str,
        images_folder_target: str,
    ) -> None:
        """Build the folder structure and copy files

        Args:
            images_folder (str): Folder where the images are located
            images_folder_target (str): Folder where the images will be copied
        """
        if not exists(images_folder_target):
            mkdir(images_folder_target, 0o755)

        for label in tqdm(self.df["label"].value_counts().index):
            if not exists(f"{images_folder_target}/{label}"):
                mkdir(f"{images_folder_target}/{label}", 0o755)

            for image in self.df[self.df["label"] == label].image_lien:
                source = f"{images_folder}/{image}"
                dest = f"{images_folder_target}/{label}/{image}"

                try:
                    if exists(source) and not exists(dest):
                        copyfile(source, dest)
                except FileNotFoundError as e:
                    print(e)
