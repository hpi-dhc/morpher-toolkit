from abc import ABC
import logging


class Job(ABC):
    """Class is used when Worker Framework is not available, i.e., locally """
    def __init__(self, config=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input = {}
        self.output = {}
        self.config = config
