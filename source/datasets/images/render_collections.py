from glob import glob
from random import sample
import matplotlib.pyplot as plt
import cv2


class CollectionFromFolder:
    def __init__(self, folder, sample_size=10, title="Class Image Sample"):
        self.folder = folder
        self.folder_name = folder.split("/")[-1]
        self.sample_size = sample_size
        self.img_row_count = self.sample_size // 2
        self.title = title
        self.images = sample(glob(f"{self.folder}/*"), sample_size)

    def render(self):
        self.fig, axs = plt.subplots(2, self.img_row_count, figsize=(30, 10))
        self.fig.suptitle(f"{self.folder_name} {self.title}", fontsize=20)
        axs = axs.flatten()
        for img, ax in zip(self.images, axs):
            img_bgr = cv2.imread(img)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            filename = img.split("/")[-1]
            ax.set_title(filename)
            ax.axis("off")

        self.fig.tight_layout()

        return self

    def save(self, filename):
        self.fig.savefig(filename)


class CollectionFromDataFrame:
    def __init__(
        self,
        dataframe,
        label_name="label",
        feature_name="label",
        sample_size=10,
        title="Class Image Sample",
    ):
        self.df = dataframe
        self.folder_name = folder.split("/")[-1]
        self.sample_size = sample_size
        self.img_row_count = self.sample_size // 2
        self.title = title
        self.label_name = label_name
        self.feature_name = feature_name

    def render(self, label=None):
        if label == None or label not in list(self.df[self.label_name].unique()):
            raise ValueError(f"Invalid label name {label}")

        sample = self.df[self.df[self.label_name] == label].sample(self.sample_size)
        self.images = list(sample[self.feature_name])

        self.fig, axs = plt.subplots(2, self.img_row_count, figsize=(30, 10))
        self.fig.suptitle(f"{self.folder_name} {self.title}", fontsize=20)
        axs = axs.flatten()
        for img, ax in zip(self.images, axs):
            img_bgr = cv2.imread(img)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            filename = img.split("/")[-1]
            ax.set_title(filename)
            ax.axis("off")

        self.fig.tight_layout()

        return self

    def save(self, filename):
        self.fig.savefig(filename)
