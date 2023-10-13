import logging


class Logger(object):
    """Creates and configure a training logger

    Args:
        object (_type_): _description_
    """

    def __init__(self, output_report, level=logging.DEBUG, logger_name="cnn"):
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
    def logger(self):
        return self.__logger
