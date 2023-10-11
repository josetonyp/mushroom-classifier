import hashlib

from tinydb import Query, TinyDB


class TrainingRecords(object):
    """Records the training data in a CSV file

    Data Structure:

    * starts_at: datetime
    * ends_at: datetime
    * base_model: str
    * architecture: str
    * accuracy: float
    """

    def __init__(self, project_folder: str) -> None:
        self.__project_folder = project_folder
        self.__db_file = f"{self.__project_folder}/training_record.json"
        self.__db = TinyDB(self.__db_file)

    @property
    def db_file(self) -> str:
        return self.__db_file

    @property
    def db(self) -> TinyDB:
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
        _id = hashlib.shake_256(
            f"{starts_at}_{base_model}_{architecture}".encode("utf-8")
        ).hexdigest(32)

        Record = Query()
        if self.db.search(Record.id == _id) == []:
            self.db.insert(
                {
                    "id": _id,
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
                Record.id == _id,
            )

        return len(self.db.all())

    def get_last_trained(self, architecture, model):
        R = Query()
        return self.db.search(
            (R.architecture == architecture) & (R.base_model == model)
        )[-1]
