import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from matplotlib.colors import LinearSegmentedColormap

grg = LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256)


class LabelValueCounts:
    def __init__(self, label_counts, subtitle=""):
        self.label_counts = label_counts
        self.subtitle = subtitle

    def render(
        self,
        figsize=(20, 10),
    ):
        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize

        self.fig, ax = plt.subplots(figsize=figsize, dpi=200)
        sns.barplot(x=self.label_counts.index.values, y=self.label_counts.values, ax=ax)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height()}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=11,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("Species")
        ax.set_xticklabels(self.label_counts.index.values, rotation="vertical")
        ax.set_ylabel("Counts")
        ax.set_title("SpeciesCount Plot of Genus in this dataset")

        return self

    def save(self, filename):
        self.fig.savefig(filename)
