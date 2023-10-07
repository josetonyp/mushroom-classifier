from glob import glob
import pandas as pd


class FolderDataset:
    def __init__(self, folder):
        self.folder = folder

    def build(self):
        files = glob(f"{self.folder}/**/*")
        labels = list(map(lambda x: x.split("/")[-2], files))

        self.df = pd.DataFrame(
            {"feature": files, "label": labels, "label_name": labels}
        )
        self.label_names = self.df.label_name.value_counts().index

        factorization = {
            v: i for i, v in enumerate(self.df.label_name.value_counts().index)
        }
        self.df["label"] = self.df.label.replace(factorization)

        return self

    def find_by_shape(self):
        import matplotlib.pyplot as plt

        horizontals = []
        for file in self.df.feature:
            shape = plt.imread(file).shape
            horizontals.append(shape[0] < shape[1])

        self.df["horizontal"] = horizontals

        return self

    def sample(self, count):
        return (
            self.df.groupby("label")
            .apply(lambda x: x.sample(count))
            .reset_index(drop=True)
        )
