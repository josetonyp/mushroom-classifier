import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ipdb


class RandomSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def render(self, nrows=1, ncols=1, figsize=(20, 20)):
        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize
        side_size = side_size * 0.65
        self.fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.grid(False)
        plt.axis("off")
        plt.tight_layout()
        samples_count = 1
        samplesdf = pd.DataFrame({"feature": [], "label": [], "label_name": []})

        datasetdf = self.dataset.df[self.dataset.df.horizontal]

        for label, i in zip(
            self.dataset.label_names, self.dataset.df.label.value_counts().index
        ):
            samples = datasetdf[datasetdf.label == i].sample(samples_count)
            samples["label_name"] = np.repeat(label, samples_count)
            samplesdf = pd.concat([samplesdf, samples])

        samplesdf = samplesdf.reset_index()
        print(samplesdf.shape)
        print(samplesdf)
        rows = min([samplesdf.shape[0] // 2, samplesdf.shape[0] // 3, 2])
        print(rows)
        for row in samplesdf.iterrows():
            im = plt.imread(row[1].feature)
            ax = plt.subplot(rows, (samplesdf.shape[0] // rows) + 1, row[0] + 1)
            ax.imshow(im)
            ax.set_title(row[1].label_name, fontsize=70)
            ax.axis("off")

        return self

    def save(self, filename):
        self.fig.savefig(filename)
