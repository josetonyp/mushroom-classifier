from __future__ import annotations
from os import makedirs
from os.path import exists
from datetime import datetime


class Folders:
    """Builds and Creates a folder where to document the exuction of the training"""

    def __init__(
        self,
        output_folder: str,
        project_name: str,
        architecture: str,
        base_models: list,
    ) -> None:
        self.__output_folder = output_folder
        self.__project_name = project_name
        self.__architecture = architecture
        self.__target_folder = None

    def build(
        self,
    ) -> None:
        """Builds the base folder if not created base on training params"""
        if not exists(self.__output_folder):
            makedirs(self.__output_folder)

        project_folder = f"{self.__output_folder}/{self.__project_name}"
        if not exists(project_folder):
            makedirs(project_folder)

        self.__target_folder = f"{project_folder}/{self.__architecture}"
        if not exists(self.__target_folder):
            makedirs(self.__target_folder)

        return self

    def create(self, base_model_name: str) -> str:
        """Creates the ouput training folder where to document the model training

        Args:
            base_model_name (str): Base model choosen to run the training

        Returns:
            str: Training model folder path
        """
        model_folder = f"{self.__target_folder}/{base_model_name}"
        if not exists(model_folder):
            makedirs(model_folder)

        version = datetime.now().strftime("%Y%m%d%H%M%S")
        training_folder = f"{model_folder}/{version}"
        makedirs(training_folder)

        print(f"Output folder {training_folder} created")

        return training_folder