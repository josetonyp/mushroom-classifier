from glob import glob
from os import makedirs
from os.path import exists
from shutil import move

import pandas as pd
from sklearn.model_selection import train_test_split


class FolderTrainTestSplit:
    def __init__(self, folder, target_folder):
        self.folder = folder
        self.target_folder = target_folder
        self.files = glob(f"{self.folder}/**/*")

    def build_dataset(self):
        labels = []
        files = []
        for file in self.files:
            label = file.split("/")[2]
            labels += [label]
            files += [file]

        self.dataset = pd.DataFrame({"file": files, "label": labels})
        return self

    def makedirs(self):
        if not exists(self.target_folder):
            makedirs(self.target_folder)

        for p in ["train", "test", "valid"]:
            if not exists(f"{self.target_folder}/{p}"):
                makedirs(f"{self.target_folder}/{p}")
            for label in self.dataset.label.value_counts().index:
                if not exists(f"{self.target_folder}/{p}/{label}"):
                    makedirs(f"{self.target_folder}/{p}/{label}")

        return self

    def split(self, test_size=0.2, valid_size=0.2):
        train, valid = train_test_split(
            self.dataset, stratify=self.dataset.label, test_size=valid_size
        )
        train, test = train_test_split(
            train,
            stratify=train.label,
            test_size=test_size,
        )

        for file, label in zip(train.file, train.label):
            fname = file.split("/")[-1]
            move(file, f"{self.target_folder}/train/{label}/{fname}")

        for file, label in zip(test.file, test.label):
            fname = file.split("/")[-1]
            move(file, f"{self.target_folder}/test/{label}/{fname}")

        for file, label in zip(valid.file, valid.label):
            fname = file.split("/")[-1]
            move(file, f"{self.target_folder}/valid/{label}/{fname}")

        return self
