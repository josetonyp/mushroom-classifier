from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

grg = LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256)


class LabelValueCounts:
    """
    A class that represents the value counts of a categorical variable.

    Attributes:
        label_counts (pandas.Series): The value counts of the categorical
        variable.

    Methods:
        render: Renders a bar plot of the value counts.

    Example:
        >>> data = pd.read_csv('data.csv')
        >>> label_counts = data['species'].value_counts()
        >>> lvc = LabelValueCounts(label_counts)
        >>> lvc.render()
    """

    def __init__(self, label_counts: pd.Series) -> None:
        self.label_counts = label_counts

    def render(
        self,
        figsize: tuple = (20, 10),
    ) -> LabelValueCounts:
        """
        Renders a bar plot of the value counts.

        Args:
            figsize (tuple): The size of the figure in inches. Defaults
            to (20,10).

        Returns:
            self: The LabelValueCounts object.
        """
        plt.rcParams["font.family"] = "Optima LT Std"

        self.fig, ax = plt.subplots(figsize=figsize, dpi=200)
        sns.barplot(
            x=self.label_counts.index.values,
            y=self.label_counts.values,
            ax=ax,
        )

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
        ax.set_xticklabels(
            self.label_counts.index.values,
            rotation="vertical",
        )
        ax.set_ylabel("Counts")
        ax.set_title("SpeciesCount Plot of Genus in this dataset")

        return self

    def save(self, filename: str) -> None:
        """
        Saves the current figure to a file with the given filename.

        Args:
            filename (str): The name of the file to save the figure to.
        """
        self.fig.savefig(filename)
