import typing as tp

from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import Constraint
from stellacode.surface.imports import get_plasma_surface


class DistanceCost(AbstractCost):
    """Non linear penalization of the distance to the plasma (lower bound)"""

    Sp: tp.Any
    constraint: Constraint = Constraint(limit=0.0, distance=0.2, weight=1.0, minimum=True)

    @classmethod
    def from_config(
        cls,
        config,
        Sp=None,
    ):
        if Sp is None:
            Sp = get_plasma_surface(config)()
        min_val = float(config["optimization_parameters"]["d_min_hard"])
        min_soft = float(config["optimization_parameters"]["d_min_soft"])
        return cls(
            Sp=Sp,
            constraint=Constraint(
                limit=min_val,
                distance=min_soft - min_val,
                weight=float(config["optimization_parameters"]["d_min_penalization"]),
            ),
        )

    def cost(self, S, results: Results = Results()):
        dist_min = S.get_distance(self.Sp.xyz).min((-1, -2))
        dist_max = S.get_distance(self.Sp.xyz).max((-1, -2))
        loss = self.constraint.barrier(dist_min)
        cost = np.einsum("ij,ij->", loss, S.ds / S.npts)

        return cost, {"min_distance": np.min(dist_min), "mean_min_distance": np.mean(dist_min), "max_distance": np.max(dist_max), "mean_max_distance": np.mean(dist_max),"cost_distance": cost}, results, S