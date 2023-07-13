from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.costs.auxi import f_e


class AreaCost(AbstractCost):
    """Non linear penalization on the area (upper bound)"""

    c0: float
    c1: float

    @classmethod
    def from_config(cls, config, Sp=None):
        return cls(
            c0=float(config["optimization_parameters"]["perim_c0"]),
            c1=float(config["optimization_parameters"]["perim_c1"]),
        )

    def cost(self, S):
        area = np.sum(S.ds) / S.npts
        area_cost = f_e(self.c0, self.c1, area)
        aux_dic = {}
        aux_dic["area"] = area
        return area_cost, aux_dic
