from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Base:
    """Dataset common functions for process, filter and manage a DataSets"""

    def original(self) -> pd.DataFrame:
        """Return the original DataSet

        Returns:
            pd.DataFrame: DataSet
        """
        return self.__df

    def data(self) -> pd.DataFrame:
        """Return the processed DataSet

        Returns:
            pd.DataFrame: DataSet
        """
        self.df["label"] = self.df["label"].astype(str)

        return self.df

    def find_n_top_labels(self, n_class: int = 5) -> Base:
        """Find the top N labels and filters the dataset to only those labels

        Args:
            n_class (int, optional): Classes to filter. Defaults to 5.

        Returns:
            Base: Dataset
        """
        top_counts = self.df["label"].value_counts()[:n_class].index
        self.df = self.df[self.df["label"].isin(list(top_counts))]

        return self

    def sample(self, count: int) -> Base:
        """Takes a sample group by label of count elements

        Args:
            count (int): Elements to sample per label

        Returns:
            Base: Dataset
        """
        self.df = (
            self.df.groupby("label")
            .apply(lambda x: x.sample(count))
            .reset_index(drop=True)
        )
        return self

    def build_image_shape(self) -> Base:
        """Process all images and computes it orientation
        in a "horizontal" feature.

        Returns:
            Base: DataSet
        """
        horizontals = []
        for file in self.df.feature:
            shape = plt.imread(file).shape
            horizontals.append(shape[0] < shape[1])

        self.df["horizontal"] = horizontals

        return self

    def horizontals(self) -> Base:
        """Filters the DataSet for only horizontal images

        Returns:
            Base: _description_
        """
        self.df = self.df[self.df["horizontal"]]

        return self

    def factorize_labels(self) -> Base:
        """Converts labels to integer orders by value counts

        Returns:
            Base: DataSet
        """
        n_class = len(self.df.label.value_counts())
        label_names = self.label_statistics[:n_class].index
        # Resets the label ids for training based on their frequency
        labels = self.df.label.value_counts().index
        factorization = {v: i for i, v in enumerate(labels)}
        self.df["label"] = self.df.label.replace(factorization)

        self.label_names = label_names

        self.selected_label_statistics = pd.DataFrame(
            {
                "label": label_names,
                "count": self.df.label.value_counts().values,
            }
        ).set_index("label")

        return self

    def split_sample(
        self,
        valid_size: float = 0.2,
        test_size: float = 0.2,
    ) -> tuple:
        """Splits the Dataset into train, valid and test subsets

        Args:
            valid_size (float, optional): Validation size in %.
            Defaults to 0.2.
            test_size (float, optional): Testing size in %.
            Defaults to 0.2.

        Returns:
            tuple: Subset of data.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.df["feature"],
            self.df["label"],
            stratify=self.df["label"],
            test_size=test_size,
        )

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            stratify=y_train,
            test_size=valid_size,
        )

        t = {}
        t["feature"] = X_train
        t["label"] = y_train  # to_categorical
        X_train = pd.DataFrame(t)

        t = {}
        t["feature"] = X_valid
        t["label"] = y_valid  # to_categorical
        X_valid = pd.DataFrame(t)

        t = {}
        t["feature"] = X_test
        t["label"] = y_test  # to_categorical
        X_test = pd.DataFrame(t)

        return X_train, X_valid, X_test

    def shuffle(self) -> Base:
        """Shuffles the DataSet

        Returns:
            Base: DataSet
        """
        self.df = shuffle(self.df)
        return self

    def save_label_statistics(self, folder: str) -> Base:
        """Documents Dataset project statistics

        Args:
            folder (_type_): Project folder
        """
        self.label_statistics.to_csv(
            f"{folder}/label_statistics.csv",
        )
        self.selected_label_statistics.to_csv(
            f"{folder}/selected_label_statistics.csv",
        )
        return self

    def downsample_to_equal(self, sample_count: int | None = None) -> Base:
        """Reduces the sample count to a given number per class

        Args:
            sample_count (int | None, optional): Numer of samples per classs.
            Defaults to None.

        Returns:
            Base: _description_
        """
        if sample_count is None:
            sample_count = min(self.df["label"].value_counts().values)

        groups = self.df.groupby("label")

        groups_list = []
        for g in groups.groups:
            groups_list.append(groups.get_group(g).sample(n=sample_count))

        self.df = pd.concat(groups_list)

        return self
