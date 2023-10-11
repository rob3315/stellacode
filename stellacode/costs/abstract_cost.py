import configparser
import typing as tp

from jax.typing import ArrayLike
from pydantic import BaseModel, Extra


class Results(BaseModel):
    """
    Results that are transmitted from costs to costs

    Args:
        * j_3d: 3d current
        * j_s: surfacic current
        * phi_mn: current weights
        * phi_mn_wnet_cur: current weights with net poloidal or toroidal currents
        * bnorm_plasma_surface: normal magnetic field onto the plasma surface
        * b_plasma_surface: magnetic field onto the plasma surface
    """

    j_3d: tp.Optional[ArrayLike] = None
    j_s: tp.Optional[ArrayLike] = None
    phi_mn: tp.Optional[ArrayLike] = None
    phi_mn_wnet_cur: tp.Optional[ArrayLike] = None
    bnorm_plasma_surface: tp.Optional[ArrayLike] = None
    b_plasma_surface: tp.Optional[ArrayLike] = None

    model_config = dict(arbitrary_types_allowed=True)


class AbstractCost(BaseModel):
    """Interface for any cost"""

    model_config = dict(arbitrary_types_allowed=True)

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
