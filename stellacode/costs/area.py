from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import inverse_barrier


class AreaCost(AbstractCost):
    """Non linear penalization on the area (upper bound)"""

    max_val: float
    distance: float = 1.0
    weight: float = 1.0

    @classmethod
    def from_config(cls, config, Sp=None):
        c0 = float(config["optimization_parameters"]["perim_c0"])
        c1 = float(config["optimization_parameters"]["perim_c1"])
        return cls(
            distance=c1 - c0,
            max_val=c1,
        )

    def cost(self, S, results: Results = Results()):
        area = S.area
        area_cost = inverse_barrier(val=-area, min_val=-self.max_val, distance=self.distance, weight=self.weight)

        aux_dic = {}
        aux_dic["area"] = area
        return area_cost, aux_dic, results, S
