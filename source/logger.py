import logging


class Logger(object):
    """Creates and configure a training logger"""

    def __init__(
        self,
        output_report: str,
        level: int = logging.DEBUG,
        logger_name: str = "cnn",
    ):
        """
        Initializes a new Logger object.

        Args:
            output_report (str): The path to the log file.
            level (int): The logging level (default: logging.DEBUG).
            logger_name (str): The name of the logger (default: "cnn").
        """
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(level)
        fh = logging.FileHandler(output_report)
        fh.setLevel(level)
        self.__logger.addHandler(fh)

    @property
    def logger(self) -> logging.Logger:
        """
        Returns the logger object associated with this instance.
        """
        return self.__logger
