import matplotlib.pyplot as plt
import numpy as np
import re, itertools


class ClassificationReport:
    def __init__(self, report, model_name, subtitle=""):
        self.report = report
        self.model_name = model_name
        self.subtitle = subtitle

    def render(self, title="Classification report ", cmap="RdBu", class_names=None):
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

        self.fig, ax = plt.subplots(figsize=(15, 10))

        self.fig.suptitle(
            f"Classification Report Model {self.model_name.upper()}", fontsize=24
        )
        if self.subtitle != "":
            ax.set_title(self.subtitle, fontsize=15)

        matrix = np.array(matrix)
        values = matrix[:, 1:]

        im = ax.imshow(values, interpolation="nearest", cmap="Blues")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Classes")

        xtick_marks = np.arange(len(values[0]))
        ax.set_xticks(xtick_marks, ["Precision", "Recall", "F1-score", "support"])

        ytick_marks = np.arange(len(matrix))
        ax.set_yticks(ytick_marks, [int(x) for x in matrix[:, 0]])

        for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
            im.axes.text(
                j,
                i,
                values[i, j],
                horizontalalignment="center",
                color="white" if values[i, j] > (values.max() / 2) else "black",
            )

        return self

    def save(self, filename):
        self.fig.savefig(filename)
