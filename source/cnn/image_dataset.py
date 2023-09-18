import pandas as pd
import os, shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class ImageDataSet:
    """Loads, process, filter and manage a DataSet in order to prepare it for training or prediction"""

    def __init__(self, filename, image_feature, label_feature, sample_count=None):
        self.filename = filename
        self.image_feature = image_feature
        self.label_feature = label_feature
        self.sample_count = sample_count

    def load(self):
        """Loads and extracts the image files and labels"""
        df = pd.read_csv(self.filename)
        self.df = pd.DataFrame(
            {
                "feature": df[self.image_feature],
                "label": df[self.label_feature],
            }
        )
        return self

    def find_n_top_labels(self, n_class=5):
        top_counts = self.df["label"].value_counts()[:n_class].index
        self.df = self.df[self.df["label"].isin(list(top_counts))]

        return self

    def downsample_to_equal(self):
        if self.sample_count == None:
            self.sample_count = min(self.df["label"].value_counts().values)

        groups = self.df.groupby("label")

        groups_list = []
        for g in groups.groups:
            groups_list.append(groups.get_group(g).sample(n=self.sample_count))

        self.df = pd.concat(groups_list)

        return self

    def data(self):
        self.df["label"] = self.df["label"].astype(str)

        return self.df

    def split_sample(self, test_size=0.2):
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.df["feature"],
            self.df["label"],
            stratify=self.df["label"],
            test_size=test_size,
        )

        t = {}
        t["feature"] = X_train
        t["label"] = y_train  # to_categorical
        X_train = pd.DataFrame(t)
        X_train["label"] = to_categorical(X_train.label)
        t = {}
        t["feature"] = X_valid
        t["label"] = y_valid  # to_categorical
        X_valid = pd.DataFrame(t)
        X_valid["label"] = to_categorical(X_valid.label)

        return X_train, X_valid

    def build_folder(self, images_folder, images_folder_target):
        if not os.path.exists(images_folder_target):
            os.mkdir(images_folder_target, 0o755)

        for label in tqdm(self.df["label"].value_counts().index):
            if not os.path.exists(f"{images_folder_target}/{label}"):
                os.mkdir(f"{images_folder_target}/{label}", 0o755)

            for image in self.df[self.df["label"] == label].image_lien:
                source = f"{images_folder}/{image}"
                dest = f"{images_folder_target}/{label}/{image}"

                try:
                    if os.path.exists(source) and not os.path.exists(dest):
                        shutil.copyfile(source, dest)
                except FileNotFoundError as e:
                    pass
                    print(e)
