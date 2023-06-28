import configparser
from abc import ABC, abstractmethod
from pydantic import BaseModel, Extra


class AbstractCost:#(BaseModel):
    """Interface for any cost

    :param S: a surface
    :type S: for now only Surface_Fourier are supported
    """

    def cost(self, S):
        raise NotImplementedError

    # @abstractmethod
    @classmethod
    def from_config(cls, config):
        pass

    @classmethod
    def from_config_file(cls, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        return cls.from_config(config)
