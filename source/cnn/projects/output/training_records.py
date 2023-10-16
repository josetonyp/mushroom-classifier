import hashlib

from tinydb import Query, TinyDB


class TrainingRecords:
    """Records the training data in a CSV file"""

    def __init__(self, project_folder: str) -> None:
        """Initializes the TrainingRecords object with the given project
        folder path.

        Args:
            project_folder (str): The path to the project folder.
        """
        self.__project_folder = project_folder
        self.__db_file = f"{self.__project_folder}/training_record.json"
        self.__db = TinyDB(self.__db_file)

    @property
    def db_file(self) -> str:
        """
        Returns the path to the database file used for storing
        training records.
        """
        return self.__db_file

    @property
    def db(self) -> TinyDB:
        """
        Returns the TinyDB instance used for storing training records.
        """
        return self.__db

    def save(
        self,
        project_folder: str,
        starts_at: str,
        base_model: str,
        architecture: str,
        epochs: int,
        batch_size: int,
        n_class: int,
        dataset_size: int,
        accuracy: float = -1.0,
        ends_at: str = "",
    ) -> int:
        """Upsert a training record in the DB

        Args:
            starts_at (str): Datetime string representing the starting time
            base_model (str): Base model ex: vgg16
            architecture (str): Architecture ex: a or b
            accuracy (float, optional): Prediction accuracy. Defaults to -1.0.
            ends_at (str, optional): Datetime string representing the
            ending time. Defaults to "".

        Returns:
            int: DB records count
        """
        uuid = hashlib.shake_256(
            f"{starts_at}_{base_model}_{architecture}".encode("utf-8")
        ).hexdigest(32)

        record = Query()
        if self.db.search(record.id == uuid) == []:
            self.db.insert(
                {
                    "id": uuid,
                    "project_folder": project_folder,
                    "starts_at": starts_at,
                    "base_model": base_model,
                    "architecture": architecture,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "n_class": n_class,
                    "datasets_size": dataset_size,
                }
            )
        else:
            self.db.update(
                {
                    "accuracy": accuracy,
                    "ends_at": ends_at,
                },
                record.id == uuid,
            )

        return len(self.db.all())

    def get_last_trained(self, architecture: str, model: str) -> dict:
        """
        Returns a dictionary containing the last trained record for a given
        architecture and model.

        Args:
            architecture (str): The name of the architecture.
            model (str): The name of the base model.

        Returns:
            dict: A dictionary containing the last trained record for the
            given architecture and model.
        """

        r = Query()
        return self.db.search(
            (r.architecture == architecture) & (r.base_model == model)
        )[-1]
