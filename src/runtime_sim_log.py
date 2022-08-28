import logging
from pathlib import Path


class Singleton(type):

    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super().__call__(*args, **kwargs)
        return cls._instance[cls]


class RuntimeLogging(metaclass=Singleton):
    def __init__(self, filedir: Path = None):
        if filedir is None:
            self.info_filename = Path("sim_logging.txt")
        else:
            self.info_filename = filedir / "sim_logging.txt"

        self.START_LOGGING = False
        self.logger = self.init_logger()

    def init_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        info_handler = logging.FileHandler(filename=self.info_filename)
        info_handler.setLevel(logging.INFO)
        info_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        info_handler.setFormatter(info_formatter)

        logger.addHandler(info_handler)
        return logger

    def info(self, *args, **kwargs):
        if self.START_LOGGING:
            self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        if self.START_LOGGING:
            self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self.START_LOGGING:
            self.logger.error(*args, **kwargs)
