import configparser
import typing as tp

from jax.typing import ArrayLike
from pydantic import BaseModel, Extra


class Results(BaseModel):
    j_3d: tp.Optional[ArrayLike] = None
    j_s: tp.Optional[ArrayLike] = None
    phi_mn: tp.Optional[ArrayLike] = None
    phi_mn_wnet_cur: tp.Optional[ArrayLike] = None
    bnorm_plasma_surface: tp.Optional[ArrayLike] = None
    b_plasma_surface: tp.Optional[ArrayLike] = None
    b_plasma: tp.Optional[ArrayLike] = None
    b_coil: tp.Optional[ArrayLike] = None

    class Config:
        arbitrary_types_allowed = True


class AbstractCost(BaseModel):
    """Interface for any cost"""

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow  # allow extra fields

    def cost(self, S, results: Results = Results()):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, Sp=None):
        raise NotImplementedError

    @classmethod
    def from_config_file(cls, config_file, Sp=None):
        config = configparser.ConfigParser()
        config.read(config_file)

        return cls.from_config(config, Sp=Sp)
