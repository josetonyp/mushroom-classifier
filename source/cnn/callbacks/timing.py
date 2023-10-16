from timeit import default_timer as timer

from tensorflow.keras.callbacks import Callback


class TimingCallback(Callback):
    """
    A Keras callback that records the time taken for each epoch during
    training.

    Attributes:
        logs (list): A list to store the time taken for each epoch.
    """

    def __init__(self, logs: dict = {}) -> None:
        self.logs = []

    def on_epoch_begin(self, epoch: int, logs: dict = {}) -> None:
        """
        Called at the beginning of each epoch in Keras training.

        Parameters:
            epoch (int): The current epoch number.
            logs (dict): Dictionary of logs, including the current
            loss and metrics.
        Returns:
            None
        """
        self.starttime = timer()

    def on_epoch_end(self, epoch: int, logs: dict = {}) -> None:
        """
        Called at the end of each epoch in Keras training.

        Parameters:
            epoch (int): The current epoch number.
            logs (dict): Dictionary of logs, including the current
            loss and metrics.
        Returns:
            None
        """
        self.logs.append(timer() - self.starttime)
