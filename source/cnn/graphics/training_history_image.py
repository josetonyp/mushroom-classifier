import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


class TrainingHistoryImage:
    def __init__(self, history_file):
        self.history = pd.read_csv(history_file)

    def render(self):
        self.fig = plt.figure(figsize=(12, 4))
        ax1, ax2 = self.fig.subplots(1, 2)

        ax1.plot(self.history.index, self.history["loss"])
        ax1.plot(self.history.index, self.history["val_loss"])
        ax1.set_title("Model loss by epoch")
        ax1.set_ylabel("loss")
        ax1.set_xlabel("epoch")
        ax1.legend(["train", "test"], loc="right")

        ax2.plot(self.history.index, self.history["accuracy"])
        ax2.plot(self.history.index, self.history["val_accuracy"])
        ax2.set_title("Model accuracy by epoch")
        ax2.set_ylabel("accuracy")
        ax2.set_xlabel("epoch")
        ax2.legend(["train", "test"], loc="right")

        return self

    def save(self, filename):
        self.fig.savefig(filename)
