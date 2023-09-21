from os import makedirs
from os.path import exists
from glob import glob
from datetime import datetime

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from source.cnn.architectures.factory import Factory as ArchFactory
from source.cnn.bases.factory import Factory as BaseModelFactory
from source.cnn.generators.folder_generator import FolderGenerator
from source.cnn.generators.image_folder_generator import ImageFolderGenerator

from source.cnn.trainer import Trainer
from source.cnn.predictor import Predictor
from source.logger import Logger as CNNLogger

from source.datasets.images.render_collections import CollectionFromFolder


class Training:
    def __init__(self, project, base_name, project_folder):
        self.project = project
        self.base_model_name = base_name
        self.project_folder = project_folder
        self.logger = (
            CNNLogger(
                f"{self.project_folder}/report.txt",
                logger_name=f"CNN Images by folder model {self.base_model_name}",
            )
        ).get_logger()

    def train(self, architecture="a", base_layer_train=0):
        self.base = BaseModelFactory.build(
            self.base_model_name,
            self.project.file_size,
            base_layer_train=base_layer_train,
        )

        self.logger.info(f"Training with Architecture {architecture}")
        self.logger.info(f"Training Base layers {base_layer_train} onwards")
        self.logger.info(f"Training with Generator ImageFolderGenerator")

        self.model = ArchFactory.build(architecture)(
            self.base.model(), self.project.n_class, file_size=self.project.file_size
        ).build()
        self.logger.info(self.model.summary(print_fn=lambda x: self.logger.info(x)))

        gen = ImageFolderGenerator(self.base.preprocess_input_method())
        train_generator = gen.generator(
            "train",
            f"{self.project.images_folder}/train",
            target_size=self.project.file_size,
            batch_size=self.project.batch_size,
        )
        valid_generator = gen.generator(
            "valid",
            f"{self.project.images_folder}/valid",
            target_size=self.project.file_size,
            batch_size=self.project.batch_size,
        )

        self.trainer = Trainer(
            self.model,
            n_class=self.project.file_size,
            batch_size=self.project.batch_size,
            target_file_size_shape=self.project.file_size,
            logger=self.logger,
        )

        self.trainer.train(train_generator, valid_generator)
        self.trainer.save(f"{self.project_folder}/model.keras")
        # self.trainer.history(f"{self.project_folder}/history.json")
        return self

    def predict(self):
        self.base = BaseModelFactory.build(self.base_model_name, self.project.file_size)
        gen = ImageFolderGenerator(self.base.preprocess_input_method())
        generator = gen.generator(
            "valid",
            f"{self.project.images_folder}/test",
            target_size=self.project.file_size,
            batch_size=self.project.batch_size,
        )

        predictor = Predictor(
            n_class=self.project.n_class,
            batch_size=self.project.batch_size,
            target_file_size_shape=self.project.file_size,
        )

        predictor.load(f"{self.project_folder}/model.keras")

        predictor.predict(generator)

        predictor.classification_report(
            to_file=f"{self.project_folder}/classification_report.txt"
        )
        predictor.confusion_matrix(
            to_file=f"{self.project_folder}/confusion_matrix.json"
        )
        return self


class ImageProject:
    def __init__(
        self,
        name,
        images_folder,
        base_models_folder="models",
        file_size=(254, 254),
        batch_size=64,
        architecture="a",
    ):
        self.name = name
        self.models_folder = f"{base_models_folder}/{name}"
        self.images_folder = images_folder
        self.file_size = file_size
        self.batch_size = batch_size
        self.architecture = architecture

        if not exists(self.models_folder):
            makedirs(self.models_folder)

        if (
            not exists(f"{self.images_folder}/train")
            or not exists(f"{self.images_folder}/test")
            or not exists(f"{self.images_folder}/valid")
        ):
            raise ValueError(
                f"Invalid {self.images_folder} configuration. Separete images into train, valid and test subfolders"
            )

        train_n_class = len(glob(f"{self.images_folder}/train/*"))
        test_n_class = len(glob(f"{self.images_folder}/test/*"))
        valid_n_class = len(glob(f"{self.images_folder}/valid/*"))

        if (
            train_n_class != test_n_class
            or valid_n_class != test_n_class
            or train_n_class != valid_n_class
        ):
            raise ValueError(
                f"Invalid number of classes per folder train: {train_n_class} test: {test_n_class} valid: {valid_n_class}"
            )
        self.n_class = train_n_class

    def document(self):
        for folder in glob(f"{self.images_folder}/train/*"):
            label = folder.split("/")[-1]

            CollectionFromFolder(folder).render().save(
                f"{self.models_folder}/{label}_sample_training_collection.jpg"
            )

        for folder in glob(f"{self.images_folder}/test/*"):
            label = folder.split("/")[-1]

            CollectionFromFolder(folder).render().save(
                f"{self.models_folder}/{label}_sample_test_collection.jpg"
            )

        for folder in glob(f"{self.images_folder}/valid/*"):
            label = folder.split("/")[-1]

            CollectionFromFolder(folder).render().save(
                f"{self.models_folder}/{label}_sample_valid_collection.jpg"
            )

    def train(self, bases=[], base_layer_train=0):
        for base in bases:
            if not exists(f"{self.models_folder}/{base}"):
                makedirs(f"{self.models_folder}/{base}")

            version = datetime.now().strftime("%Y%m%d%H%M%S")
            training_folder = (
                f"{self.models_folder}/{self.architecture}/{base}/{version}"
            )
            makedirs(training_folder)

            Training(self, base, training_folder).train(
                architecture=self.architecture, base_layer_train=base_layer_train
            ).predict()
