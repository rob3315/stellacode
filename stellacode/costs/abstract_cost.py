import configparser

from pydantic import BaseModel, Extra


class AbstractCost(BaseModel):
    """Interface for any cost"""

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow  # allow extra fields

    def cost(self, S):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, Sp=None):
        raise NotImplementedError

    @classmethod
    def from_config_file(cls, config_file, Sp=None):
        config = configparser.ConfigParser()
        config.read(config_file)

        return cls.from_config(config, Sp=Sp)
