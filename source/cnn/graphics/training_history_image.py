from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")


class TrainingHistoryImage:
    """Reads a Keras training history and plots its evolution through epochs

    Example:
    TrainingHistoryImage(data, title) \
        .render(figsize=figsize) \
        .save(f"{file_name_tag}.jpg")
    """

    def __init__(self, data: pd.DataFrame, title: str):
        self.__data = data
        self.__title = title

    def render(self, figsize: tuple = (15, 5)) -> TrainingHistoryImage:
        """Plots a Training History

        Args:
            figsize (tuple, optional): Figure Size. Defaults to (15, 5).

        Returns:
            TrainingHistoryImage: Instance of the renderer
        """
        plt.rcParams["font.family"] = "Optima LT Std"
        side_size, _ = figsize
        side_size = side_size * 0.5

        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle(
            f"Training History {self.__title.upper()}",
            fontsize=side_size * 2,
            color="lightgrey",
        )

        ax1, ax2 = self.fig.subplots(1, 2)

        ax1.plot(self.__data.index, self.__data["loss"])
        ax1.plot(self.__data.index, self.__data["val_loss"])
        ax1.set_title("Model's Trainingl Loss by Epochs", fontsize=side_size)
        ax1.tick_params(axis="x", labelsize=side_size * 0.6)
        ax1.tick_params(axis="y", labelsize=side_size * 0.6)
        ax1.set_ylabel(
            "Loss",
            fontsize=side_size * 1.5,
            color="silver",
        )
        ax1.set_xlabel(
            "Epochs",
            fontsize=side_size * 1.5,
            color="silver",
        )
        ax1.legend(["train", "test"], loc="right")

        ax2.plot(self.__data.index, self.__data["accuracy"])
        ax2.plot(self.__data.index, self.__data["val_accuracy"])
        ax2.set_title(
            "Model's Training Accuracy by Epochs",
            fontsize=side_size,
        )
        ax2.tick_params(axis="x", labelsize=side_size * 0.6)
        ax2.tick_params(axis="y", labelsize=side_size * 0.6)
        ax2.set_ylabel(
            "Accuracy",
            fontsize=side_size * 1.5,
            color="silver",
        )
        ax2.set_xlabel(
            "Epochs",
            fontsize=side_size * 1.5,
            color="silver",
        )
        ax2.legend(["train", "test"], loc="right")

        return self

    def save(self, filename: str) -> None:
        """Save the generated figure into a file

        Args:
            filename (str): Target file name
        """
        self.fig.savefig(filename)
