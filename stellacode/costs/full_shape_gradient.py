import configparser
import logging
from time import time

import stellacode.costs.EM_shape_gradient as EM
from stellacode import np
from stellacode.costs.curvature_shape_gradient import Curvature_shape_gradient
from stellacode.costs.distance_shape_gradient import Distance_shape_gradient
from stellacode.costs.perimeter_shape_gradient import Perimeter_shape_gradient


class Full_shape_gradient:
    """Put together all the costs"""

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        self.Np = int(config["geometry"]["Np"])
        self.ntheta_coil = int(config["geometry"]["ntheta_coil"])
        self.nzeta_coil = int(config["geometry"]["nzeta_coil"])

        # Initialization of the different costs :
        self.EM = EM.EM_shape_gradient(config=config)
        self.S = self.EM.S
        self.init_param = self.S.param
        self.lst_cost = [self.EM]
        if config["optimization_parameters"]["d_min"] == "True":
            self.dist = Distance_shape_gradient(config=config)
            self.lst_cost.append(self.dist)
        if config["optimization_parameters"]["perim"] == "True":
            self.perim = Perimeter_shape_gradient(config=config)
            self.lst_cost.append(self.perim)
        if config["optimization_parameters"]["curvature"] == "True":
            self.curv = Curvature_shape_gradient(config=config)
            self.lst_cost.append(self.curv)

    def get_surface(self):
        """
        Returns the surface.
        """
        return self.S

    def get_j_3D(self):
        """
        Returns the current distribution on the surface.
        """
        return self.j_3D

    def cost(self, param):
        self.S.param = param
        tic = time()
        c, EM_cost_dic = self.lst_cost[0].cost(self.S)
        for elt in self.lst_cost[1:]:
            tic = time()
            new_cost, _ = elt.cost(self.S)
            c += new_cost
            print(elt, tic - time())

        return c
