from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import re, itertools

from functools import cached_property

from matplotlib.colors import LinearSegmentedColormap

grg = LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256)


class ClassificationReport:
    """Reads a Keras Classfication Report and Renders a confusion matrix style
    image from that report

    Example:
    ClassificationReport(data: str, title: str, label_names: list | None =label_names)\
        .render(figsize=(<int>,<int>))\
        .save("<path_to_file>")
    """

    def __init__(
        self,
        report: str,
        title: str = "",
        subtitle: str = "",
        label_names: list | None = None,
    ) -> None:
        self.__report = report
        self.__title = title
        self.__subtitle = subtitle
        self.__label_names = label_names

    def render(
        self,
        cmap: LinearSegmentedColormap = grg,
        figsize: tuple = (50, 50),
    ) -> ClassificationReport:
        """Plots scikit-learn classification report.
        Based on https://stackoverflow.com/a/31689645/395857


        Args:
            cmap (LinearSegmentedColormap, optional): Color Map. Defaults to LinearSegmentedColormap.from_list("rg", ["#EAEAF2", "#3174A1"], N=256).
            figsize (tuple, optional): Figure Size. Defaults to (50, 50).

        Returns:
            ClassificationReport: Instance of the renderer
        """

        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize
        side_size = side_size * 0.7
        self.fig, ax = plt.subplots(figsize=figsize)

        self.fig.suptitle(
            f"Classification Report {self.__title}",
            fontsize=side_size * 1.5,
        )
        ax.set_title(
            self.__subtitle + f"Total Accuracy: {self.accuracy}",
            fontsize=side_size * 1.5,
        )
        ax.title.set_color("dimgray")

        matrix = self.classification_matrix
        values = matrix[:, 1:]

        im = ax.imshow(values, interpolation="nearest", cmap=cmap)
        ax.set_xlabel("Metrics", fontsize=side_size * 1.5)
        ax.set_ylabel("Classes", fontsize=side_size * 1.5)

        if self.__label_names != None:
            ax.set_yticks(
                np.arange(len(matrix)),
                self.__label_names,
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
                fontsize=side_size * 1.5,
            )

        return self

    def save(self, filename: str) -> None:
        """Save the generated figure into a file

        Args:
            filename (str): Target file name
        """
        self.fig.savefig(filename, dpi=200)

    @cached_property
    def lines(self) -> list:
        """Converts the text report into a list of individual lines

        Returns:
            list: Report lines
        """
        return self.__report.split("\n")

    @cached_property
    def classification_matrix(self) -> np.ndarray:
        """Extracts classification precision, recall, f1-score and support from report
        and converst it into a renderable numpay array.

        Returns:
            np.ndarray: Report values
        """
        matrix = []

        for result in self.lines[2:]:
            if result.strip() == "":
                break
            nums = re.findall("[\d\.]+", result.strip())
            nums = [float(x) for x in nums]
            matrix += [nums]

        return np.array(matrix)

    @cached_property
    def accuracy(self) -> float:
        """Extracts the total report accuracy from the report

        Returns:
            float: Report total accuracy
        """
        accuracy = self.lines[2:]

        breking_index = 0
        for i, result in enumerate(self.lines[2:]):
            if result.strip() == "":
                breking_index = i
                break

        accuracy_line = self.lines[2:][breking_index + 1]

        nums = re.findall("[\d\.]+", accuracy_line.strip())
        return float(nums[0])
