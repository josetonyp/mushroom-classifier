import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from math import log
from matplotlib.colors import LinearSegmentedColormap

grg = LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256)


class ConfusionMatrix:
    def __init__(self, matrix, model_name, subtitle="", label_names=None):
        self.matrix = np.array(matrix)
        self.classes = range(0, len(matrix))
        self.model_name = model_name
        self.subtitle = subtitle
        self.label_names = label_names

    def render(
        self,
        cmap=grg,
        figsize=(50, 50),
    ):
        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize
        side_size = side_size * 0.65
        self.fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)
        self.fig.suptitle(
            f"Confusion Matrix {self.model_name.upper()}",
            fontsize=side_size * 1.5,
        )

        if self.subtitle != "":
            ax.set_title(self.subtitle, fontsize=side_size * 1.5)

        im = ax.imshow(self.matrix, interpolation="nearest", cmap=cmap)
        self.fig.colorbar(im, ax=ax)
        tick_marks = np.arange(len(self.classes))
        if self.label_names != None:
            ax.set_xticks(
                tick_marks,
                self.label_names,
                rotation="vertical",
                fontsize=side_size * 1.5,
            )
            ax.set_yticks(tick_marks, self.label_names, fontsize=side_size * 1.5)
        else:
            ax.set_xticks(tick_marks, tick_marks, fontsize=side_size * 1.5)
            ax.set_yticks(tick_marks, tick_marks, fontsize=side_size * 1.5)

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
                fontsize=side_size * 1.5,
            )

        ax.set_ylabel("True Labels", fontsize=side_size * 1.5)
        ax.set_xlabel("Predicted Labels", fontsize=side_size * 1.5)

        return self

    def save(self, filename):
        self.fig.savefig(filename, dpi=200)
