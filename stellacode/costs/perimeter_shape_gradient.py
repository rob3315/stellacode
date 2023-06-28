import configparser
import logging

from stellacode import np
from stellacode.costs.abstract_shape_gradient import Abstract_shape_gradient
from stellacode.costs.auxi import f_e


class Perimeter_shape_gradient(Abstract_shape_gradient):
    """Non linear penalization on the perimeter (upper bound)"""

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        self.Np = int(config["geometry"]["Np"])
        self.c0 = float(config["optimization_parameters"]["perim_c0"])
        self.c1 = float(config["optimization_parameters"]["perim_c1"])

    def cost(self, S):
        perimeter = self.Np * np.sum(S.dS) / S.npts
        perim_cost = f_e(self.c0, self.c1, perimeter)
        aux_dic = {}
        aux_dic["perimeter"] = perimeter
        return perim_cost, aux_dic
