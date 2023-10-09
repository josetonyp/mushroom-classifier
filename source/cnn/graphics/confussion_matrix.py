from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import itertools

from matplotlib.colors import LinearSegmentedColormap

grg = LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256)


class ConfusionMatrix:
    """Reads a Keras Confussion Matrix JSON file and Renders a confusion matrix
    image.

    Example:
    ConfusionMatrix(data,: str title: str, label_names: list | None =label_names)\
        .render(figsize=(<int>,<int>))\
        .save("<path_to_file>")
    """

    def __init__(
        self,
        matrix: np.ndarray,
        title: str = "",
        subtitle: str = "",
        label_names: list | None = None,
    ) -> None:
        self.__matrix = np.array(matrix)
        self.__classes = range(0, len(matrix))
        self.__title = title
        self.__subtitle = subtitle
        self.__label_names = label_names

    def render(
        self,
        cmap: LinearSegmentedColormap = grg,
        figsize: tuple = (50, 50),
    ) -> ConfusionMatrix:
        """Plots the Confusion Matrix

        Args:
            cmap (LinearSegmentedColormap, optional): Color Map. Defaults to LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256).
            figsize (tuple, optional): Figure Size. Defaults to (50, 50).

        Returns:
            ConfusionMatrix: Instance of the renderer
        """
        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize
        side_size = side_size * 0.8

        self.fig, ax = plt.subplots(figsize=figsize)
        ax.grid(False)

        self.fig.suptitle(
            f"Confusion Matrix {self.__title.upper()}",
            fontsize=side_size * 1.5,
        )

        if self.__subtitle != "":
            ax.set_title(self.__subtitle, fontsize=side_size * 1.5)

        im = ax.imshow(self.__matrix, interpolation="nearest", cmap=cmap)
        cbar = self.fig.colorbar(im, ax=ax)

        ticklabs = cbar.ax.get_yticklabels()
        cbar.ax.set_yticklabels(ticklabs, fontsize=side_size * 1.5)

        tick_marks = np.arange(len(self.__classes))
        if self.__label_names != None:
            ax.set_xticks(
                tick_marks,
                self.__label_names,
                rotation="vertical",
                fontsize=side_size * 1.5,
            )
            ax.set_yticks(tick_marks, self.__label_names, fontsize=side_size * 1.5)
        else:
            ax.set_xticks(tick_marks, tick_marks, fontsize=side_size * 1.5)
            ax.set_yticks(tick_marks, tick_marks, fontsize=side_size * 1.5)

        for i, j in itertools.product(
            range(self.__matrix.shape[0]), range(self.__matrix.shape[1])
        ):
            im.axes.text(
                j,
                i,
                self.__matrix[i, j],
                horizontalalignment="center",
                color="white"
                if self.__matrix[i, j] > (self.__matrix.max() / 2)
                else "black",
                fontsize=side_size * 1.5,
            )

        ax.set_ylabel("True Labels", fontsize=side_size * 1.5)
        ax.set_xlabel("Predicted Labels", fontsize=side_size * 1.5)

        return self

    def save(self, filename: str) -> None:
        """Save the generated figure into a file

        Args:
            filename (str): Target file name
        """
        self.fig.savefig(filename, dpi=200)
