from __future__ import annotations

from source.cnn.projects.output.folders import Folders as OutputFolders
from source.datasets.image_dataset import ImageDataSet
from source.cnn.projects.training import Training
from source.logger import Logger as CNNLogger


class Project:
    """Configure and executes a training project"""

    def __init__(
        self,
        name: str,
        dataset: ImageDataSet,
        base_models,
        ouput_folder: str = "models",
        file_size: tuple = (254, 254),
        batch_size: int = 64,
        architecture: str = "a",
        epochs: int = 8,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.n_class = len(dataset.label_statistics)
        self.file_size = file_size
        self.batch_size = batch_size
        self.architecture = architecture
        self.epochs = epochs

        self.__ouput_folder = ouput_folder
        self.__base_models = base_models

    def train(self, base_layer_train: int = 0) -> list:
        """Create the training folder, executes training and predictions for each given base model

        Args:
            base_layer_train (int, optional): Number of top layers to train in the base folder. It must be negative. Defaults to 0.


        Returns:
            list: List of trained project's folders
        """
        output = OutputFolders(
            self.__ouput_folder, self.name, self.architecture, self.__base_models
        ).build()

        projects = []
        for base in self.__base_models:
            base_output_folder = output.create(base)
            projects.append(base_output_folder)
            logger = (
                CNNLogger(
                    f"{base_output_folder}/report.txt",
                    logger_name=f"CNN Images by folder model {base_output_folder}",
                )
            ).get_logger()

            Training(self, base, base_output_folder, logger).train(
                architecture=self.architecture, base_layer_train=base_layer_train
            ).predict()

        return projects
