import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


class ConfusionMatrix:
    def __init__(self, matrix, model_name, subtitle=""):
        self.matrix = np.array(matrix)
        self.classes = range(0, len(matrix))
        self.model_name = model_name
        self.subtitle = subtitle

    def render(self, figsize=(12, 12)):
        self.fig, ax = plt.subplots(figsize=figsize)
        self.fig.suptitle(
            f"Confusion Matrix Model {self.model_name.upper()}", fontsize=24
        )
        if self.subtitle != "":
            ax.set_title(self.subtitle, fontsize=15)

        im = ax.imshow(self.matrix, interpolation="nearest", cmap="Blues")
        self.fig.colorbar(im, ax=ax)
        tick_marks = np.arange(len(self.classes))
        ax.set_xticks(tick_marks, self.classes)
        ax.set_yticks(tick_marks, self.classes)

        for i, j in itertools.product(
            range(self.matrix.shape[0]), range(self.matrix.shape[1])
        ):
            im.axes.text(
                j,
                i,
                self.matrix[i, j],
                horizontalalignment="center",
                color="white"
                if self.matrix[i, j] > (self.matrix.max() / 2)
                else "black",
            )

        ax.set_ylabel("True Labels")
        ax.set_xlabel("Predicted Labels")

        return self

    def save(self, filename):
        self.fig.savefig(filename)
