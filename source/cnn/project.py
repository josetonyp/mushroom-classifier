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
from source.cnn.graphics.training_history_image import TrainingHistoryImage
from source.cnn.graphics.confussion_matrix import ConfusionMatrix
from source.cnn.graphics.classification_report import ClassificationReport


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
        """Selects, builds and train a model based on a selected Architecture

        Args:
            architecture (str, optional): CNN Model Architecture. Defaults to "a".
            base_layer_train (int, optional): Base layers to be trained. Defaults to 0.

        Returns:
            _type_: _description_
        """
        self.logger.info(f"Training with Architecture {architecture}")
        self.logger.info(f"Training Base layers {base_layer_train} onwards")
        self.logger.info(f"Training with Generator ImageFolderGenerator")

        # The relationship between Base and Architecture requires a joined factory.
        # There are Model Architectures that have a base for transferred learning
        # and there are Model Architecture that are build from scratch which use no other
        # pretrained model in their base
        #
        # Here, I seed the idea of selecting a base depending on the architecture but this
        # would require encapsulation further down the delopment
        #
        if architecture not in ["a", "b"]:
            self.base_model_name = "empty"

        self.base = BaseModelFactory.build(
            self.base_model_name,
            self.project.file_size,
            base_layer_train=base_layer_train,
        )

        self.model = ArchFactory.build(architecture)(
            self.base.model(), self.project.n_class, file_size=self.project.file_size
        ).build()
        self.logger.info(self.model.summary(print_fn=lambda x: self.logger.info(x)))

        # Build the data generators. This area can also be extended as there are a few possible options
        # to load data into the model for training puporse
        #
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
            epochs=self.project.epochs,
        )

        self.trainer.train(train_generator, valid_generator)
        self.trainer.save(f"{self.project_folder}/model.keras")
        self.trainer.save_history(f"{self.project_folder}/history.csv")

        TrainingHistoryImage(f"{self.project_folder}/history.csv").render().save(
            f"{self.project_folder}/train_history.jpg"
        )

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

        ConfusionMatrix(
            predictor.cnf_matrix,
            self.base_model_name,
            subtitle=f"{self.project.name} with architecture {self.project.architecture}",
        ).render().save(f"{self.project_folder}/confusion_matrix.jpg")

        ClassificationReport(
            predictor.class_report,
            self.base_model_name,
            subtitle=f"{self.project.name} with architecture {self.project.architecture}",
        ).render().save(f"{self.project_folder}/classification_report.jpg")

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
        epochs=8,
    ):
        self.name = name
        self.models_folder = f"{base_models_folder}/{name}"
        self.images_folder = images_folder
        self.file_size = file_size
        self.batch_size = batch_size
        self.architecture = architecture
        self.epochs = epochs

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
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            training_folder = (
                f"{self.models_folder}/{self.architecture}/{base}/{version}"
            )
            makedirs(training_folder)

            Training(self, base, training_folder).train(
                architecture=self.architecture, base_layer_train=base_layer_train
            ).predict()

    def create_model_folder(self, project, architecture, model_name):
        if not exists("models"):
            makedirs("models")

        if not exists(f"models/{project}"):
            makedirs(f"models/{project}")

        architecture_folder = f"models/{project}/{architecture}"
        if not exists(architecture_folder):
            makedirs(architecture_folder)

        model_folder = f"models/{project}/{architecture}/{model_name}"
        if not exists(model_folder):
            makedirs(model_folder)

        return model_folder
