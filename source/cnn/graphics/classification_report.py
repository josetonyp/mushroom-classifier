import matplotlib.pyplot as plt
import numpy as np
import re, itertools
from matplotlib.colors import LinearSegmentedColormap

grg = LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256)


class ClassificationReport:
    def __init__(self, report, model_name, subtitle="", label_names=None):
        self.report = report
        self.model_name = model_name
        self.subtitle = subtitle
        self.label_names = label_names

    def render(
        self,
        title="Classification report ",
        cmap=grg,
        class_names=None,
        figsize=(50, 50),
    ):
        """
        Plot scikit-learn classification report.
        Extension based on https://stackoverflow.com/a/31689645/395857
        """
        lines = self.report.split("\n")

        matrix = []
        for result in lines[2:]:
            if result.strip() == "":
                break
            nums = re.findall("[\d\.]+", result.strip())
            nums = [float(x) for x in nums]
            matrix += [nums]

        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize

        self.fig, ax = plt.subplots(figsize=(side_size, side_size))

        self.fig.suptitle(
            f"Classification Report Model {self.model_name.upper()}",
            fontsize=side_size * 1.5,
        )
        if self.subtitle != "":
            ax.set_title(self.subtitle, fontsize=side_size * 1.5)
            ax.title.set_color("darkgreen")

        matrix = np.array(matrix)
        values = matrix[:, 1:]

        im = ax.imshow(values, interpolation="nearest", cmap=cmap)
        ax.set_xlabel("Metrics", fontsize=side_size * 1.5)
        ax.set_ylabel("Classes", fontsize=side_size * 1.5)

        if self.label_names != None:
            ax.set_yticks(
                np.arange(len(matrix)),
                self.label_names,
                fontsize=side_size * 1.5,
            )
        else:
            ax.set_yticks(
                np.arange(len(matrix)),
                [int(x) for x in matrix[:, 0]],
                fontsize=side_size * 1.5,
            )

        ax.set_xticks(
            np.arange(len(values[0])),
            ["Precision", "Recall", "F1-score", "support"],
            fontsize=side_size * 1.5,
        )

        ax.grid(False)
        for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
            im.axes.text(
                j,
                i,
                values[i, j],
                horizontalalignment="center",
                color="white" if values[i, j] > (values.max() / 2) else "black",
                fontsize=side_size * 2,
            )

        return self

    def save(self, filename):
        self.fig.savefig(filename)
