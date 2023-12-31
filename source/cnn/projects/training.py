from __future__ import annotations
from datetime import datetime
from source.cnn.architectures.factory import Factory as ArchFactory
from source.cnn.bases.factory import Factory as BaseModelFactory
from source.cnn.generators.image_folder_generator import DataSetFolderGenerator
from source.cnn.predictor import Predictor
from source.cnn.projects.output.training_records import TrainingRecords
from source.cnn.trainer import Trainer

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Training(object):
    """Trains a model based on a selected Architecture"""

    def __init__(
        self,
        project: object,
        base_name: str,
        project_folder: str,
        logger: object,
    ) -> Training:
        self.project = project
        self.base_model_name = base_name
        self.project_folder = project_folder
        self.__logger = logger
        self.__record_db = TrainingRecords(self.project.project_folder_path)

        self.__record_db.save(
            self.project_folder,
            self.project_folder.split("/")[-1],
            self.base_model_name,
            self.project.architecture,
            self.project.epochs,
            self.project.batch_size,
            self.project.dataset.n_class,
            self.project.dataset.df.shape[0],
        )

    def train(self, architecture="a", base_layer_train: int = 0) -> Training:
        """Selects, builds and train a model based on a selected Architecture

        Args:
            architecture (str, optional): CNN Model Architecture.
            Defaults to "a".
            base_layer_train (int, optional): Base layers to be trained.
            Defaults to 0.

        Returns:
            Training: Instance of Training
        """
        self.__logger.info(f"Training with Architecture {architecture}")
        self.__logger.info(f"Training Base layers {base_layer_train} onwards")
        self.__logger.info("Training with Generator DataSetFolderGenerator")

        # The relationship between Base and Architecture requires a
        # joined factory. There are Model Architectures that have a base
        # for transferred learning and there are Model Architecture that
        # are build from scratch which use no other pretrained model
        # in their basee
        #
        # Here, I seed the idea of selecting a base depending on the
        # architecture but this would require encapsulation further down
        # the delopment
        #
        if architecture not in ["a", "b"]:
            self.base_model_name = "empty"

        self.base = BaseModelFactory.build(
            self.base_model_name,
            self.project.file_size,
            base_layer_train=base_layer_train,
        )

        print("Loading Base Model and Architecture")
        self.model = ArchFactory.build(architecture)(
            self.base.model(),
            self.project.n_class,
        ).build()
        self.__logger.info(
            self.model.summary(print_fn=lambda x: self.__logger.info(x)),
        )

        train, valid, self.test_dataset = self.project.dataset.split_sample()
        self.project.dataset.save_label_statistics(self.project_folder)

        print("Loading Train and Validation Image Data Generators")
        gen = DataSetFolderGenerator(self.base.preprocess_input_method())
        train_generator = gen.generator(
            train,
            target_size=self.project.file_size,
            batch_size=self.project.batch_size,
            feature_name="feature",
        )
        valid_generator = gen.generator(
            valid,
            target_size=self.project.file_size,
            batch_size=self.project.batch_size,
            feature_name="feature",
        )
        self.__logger.info(
            (
                f"Training with the following classes <label names>"
                f"{self.project.dataset.label_names}"
            )
        )
        trainer = Trainer(
            self.model,
            n_class=self.project.file_size,
            batch_size=self.project.batch_size,
            target_file_size_shape=self.project.file_size,
            logger=self.__logger,
            epochs=self.project.epochs,
        )

        trainer.train(train_generator, valid_generator)
        trainer.save(f"{self.project_folder}/model.keras")
        trainer.save_history(f"{self.project_folder}/history.csv")

        train.to_csv(f"{self.project_folder}/dataset_train.csv")
        valid.to_csv(f"{self.project_folder}/dataset_validation.csv")
        self.test_dataset.to_csv(f"{self.project_folder}/dataset_test.csv")

        return self

    def predict(self) -> Training:
        """Load model and predicts the test datasets

        Returns:
            Trainng: Instance of Training
        """
        self.base = BaseModelFactory.build(
            self.base_model_name,
            self.project.file_size,
        )

        gen = DataSetFolderGenerator(self.base.preprocess_input_method())
        print("Loading Test Image Data Generators")
        generator = gen.generator(
            self.test_dataset,
            target_size=self.project.file_size,
            batch_size=self.project.batch_size,
            feature_name="feature",
        )

        predictor = Predictor(batch_size=self.project.batch_size)
        predictor.load(f"{self.project_folder}/model.keras")

        predictor.predict(generator)

        self.__record_db.save(
            self.project_folder,
            self.project_folder.split("/")[-1],
            self.base_model_name,
            self.project.architecture,
            self.project.epochs,
            self.project.batch_size,
            self.project.dataset.n_class,
            self.project.dataset.df.shape[0],
            predictor.accuracy_score(),
            datetime.now().strftime("%Y%m%d%H%M%S"),
        )

        predictor.classification_report(
            to_file=f"{self.project_folder}/classification_report.txt"
        )
        predictor.confusion_matrix(
            to_file=f"{self.project_folder}/confusion_matrix.json"
        )

        return self
