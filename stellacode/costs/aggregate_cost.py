import configparser
from time import time

from stellacode.costs.curvature import CurvatureCost
from stellacode.costs.distance import DistanceCost
from stellacode.costs.EM_shape_gradient import EMCost
from stellacode.costs.perimeter import PerimeterCost


class AggregateCost:
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
        self.EM = EMCost(config=config)
        self.S = self.EM.S
        self.init_param = self.S.params

        self.lst_cost = [self.EM]
        if config["optimization_parameters"]["d_min"] == "True":
            self.dist = DistanceCost(config=config)
            self.lst_cost.append(self.dist)
        if config["optimization_parameters"]["perim"] == "True":
            self.perim = PerimeterCost(config=config)
            self.lst_cost.append(self.perim)
        if config["optimization_parameters"]["curvature"] == "True":
            self.curv = CurvatureCost(config=config)
            self.lst_cost.append(self.curv)

    def cost(self, **kwargs):
        self.S.update_params(**kwargs)
        tic = time()
        c, EM_cost_dic = self.lst_cost[0].cost(self.S)
        for elt in self.lst_cost[1:]:
            tic = time()
            new_cost, _ = elt.cost(self.S)
            c += new_cost
            print(elt, tic - time())

        return c
