from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from source.cnn.image_dataset import ImageDataSet


class RandomSampler:
    """Given a Pandas Dataframe renders an sampler image from all rows"""

    def __init__(self, dataset: ImageDataSet) -> None:
        self.dataset = dataset

    def render(
        self, nrows: int = 1, ncols: int = 5, figsize: tuple = (20, 20)
    ) -> RandomSampler:
        """Renders the image sampler

        Args:
            nrows (int, optional): Number of rows in the image. Defaults to 1.
            ncols (int, optional): Number of columns in the image. Defaults to 5.
            figsize (tuple, optional): Figure Size. Defaults to (20, 20).

        Raises:
            Exception: Stops render if there are more images than slotes in the image

        Returns:
            RandomSampler: Instance of the sampler
        """
        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize
        side_size = side_size * 0.65

        self.fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
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

        samplesdf = samplesdf.reset_index(drop=True)

        if samplesdf.shape[0] > nrows * ncols:
            raise Exception(
                f"Samples are larger than image canvas. Samples: {samplesdf.shape[0]}, axes: {nrows*ncols}"
            )

        for i, row in samplesdf.iterrows():
            ax = axs[i]
            im = plt.imread(row.feature)
            ax.imshow(im)
            ax.grid(False)
            ax.set_title(f"{row.label_name}\n", fontsize=side_size * 1.2)
            ax.axis("off")

        return self

    def save(self, filename: str) -> None:
        """Save the generated figure into a file

        Args:
            filename (str): Target file name
        """
        self.fig.savefig(filename, dpi=200)
