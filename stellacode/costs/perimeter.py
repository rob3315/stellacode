from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.costs.auxi import f_e


class PerimeterCost(AbstractCost):
    """Non linear penalization on the perimeter (upper bound)"""

    Np: int
    c0: float
    c1: float

    @classmethod
    def from_config(cls, config):
        return cls(
            Np=int(config["geometry"]["Np"]),
            c0=float(config["optimization_parameters"]["perim_c0"]),
            c1=float(config["optimization_parameters"]["perim_c1"]),
        )

    def cost(self, S):
        perimeter = self.Np * np.sum(S.dS) / S.npts
        perim_cost = f_e(self.c0, self.c1, perimeter)
        aux_dic = {}
        aux_dic["perimeter"] = perimeter
        return perim_cost, aux_dic
