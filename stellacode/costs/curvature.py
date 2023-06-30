from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.costs.auxi import f_e


class CurvatureCost(AbstractCost):
    """Non linear penalization on the curvature (upper bound)"""

    Np: int
    ntheta_coil: int
    nzeta_coil: int
    c0: float
    c1: float

    @classmethod
    def from_config(cls, config):
        return cls(
            Np=int(config["geometry"]["Np"]),
            ntheta_coil=int(config["geometry"]["ntheta_coil"]),
            nzeta_coil=int(config["geometry"]["nzeta_coil"]),
            c0=float(config["optimization_parameters"]["curvature_c0"]),
            c1=float(config["optimization_parameters"]["curvature_c1"]),
        )

    def cost(self, S):
        pmax, pmin = S.principles[0], S.principles[1]
        fun = np.vectorize(lambda x: f_e(self.c0, self.c1, np.maximum(x, 0.0)))
        f_pmax = fun(pmax)
        f_pmin = fun(pmin)
        cost = self.Np * np.einsum("ij,ij->", f_pmax, S.dS / S.npts)
        cost += self.Np * np.einsum("ij,ij->", f_pmin, S.dS / S.npts)
        aux_dic = {}
        aux_dic["max_curvature"] = max(np.max(pmax), np.max(-pmin))

        return cost, aux_dic
