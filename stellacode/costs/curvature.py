import configparser

from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.costs.auxi import f_e


class CurvatureCost(AbstractCost):
    """Non linear penalization on the curvature (upper bound)"""

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        self.Np = int(config["geometry"]["Np"])
        self.ntheta_coil = int(config["geometry"]["ntheta_coil"])
        self.nzeta_coil = int(config["geometry"]["nzeta_coil"])
        self.c0 = float(config["optimization_parameters"]["curvature_c0"])
        self.c1 = float(config["optimization_parameters"]["curvature_c1"])
        self.f = np.vectorize(lambda x: f_e(self.c0, self.c1, np.maximum(x, 0.0)))

    def cost(self, S):
        pmax, pmin = S.principles[0], S.principles[1]

        f_pmax = self.f(pmax)
        f_pmin = self.f(pmin)
        cost = self.Np * np.einsum("ij,ij->", f_pmax, S.dS / S.npts)
        cost += self.Np * np.einsum("ij,ij->", f_pmin, S.dS / S.npts)
        aux_dic = {}
        aux_dic["max_curvature"] = max(np.max(pmax), np.max(-pmin))

        return cost, aux_dic
