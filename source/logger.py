import logging


class Logger(object):
    def __init__(self, output_report, level=logging.DEBUG, logger_name="cnn"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(output_report)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger
