import typing as tp

from jax.typing import ArrayLike

import stellacode.tools as tools
from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.costs.auxi import f_non_linear
from stellacode.surface.imports import get_plasma_surface


class DistanceCost(AbstractCost):
    """Non linear penalization of the distance to the plasma (lower bound)"""

    num_tor_symmetry: int
    Sp: tp.Any
    rot_tensor: ArrayLike
    d_min_hard: float
    d_min_soft: float
    d_min_penalization: float

    @classmethod
    def from_config(cls, config):
        num_tor_symmetry = int(config["geometry"]["Np"])
        return cls(
            num_tor_symmetry=num_tor_symmetry,
            Sp=get_plasma_surface(config),
            rot_tensor=tools.get_rot_tensor(num_tor_symmetry),
            d_min_hard=float(config["optimization_parameters"]["d_min_hard"]),
            d_min_soft=float(config["optimization_parameters"]["d_min_soft"]),
            d_min_penalization=float(config["optimization_parameters"]["d_min_penalization"]),
        )

    def cost(self, S):
        vf = np.vectorize(lambda x: f_non_linear(self.d_min_hard, self.d_min_soft, self.d_min_penalization, x))
        T = tools.get_tensor_distance(S, self.Sp, self.rot_tensor)
        dist = np.linalg.norm(T, axis=-1)
        dist_min = np.amin(dist, axis=(0, 3, 4))
        cost = np.einsum("ij,ij->", vf(dist_min), S.dS / S.npts)

        return cost, {"min_distance": np.min(dist_min)}
