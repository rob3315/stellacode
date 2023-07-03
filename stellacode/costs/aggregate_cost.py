import configparser
from time import time

from stellacode.costs.curvature import CurvatureCost
from stellacode.costs.distance import DistanceCost
from stellacode.costs.EM_cost import EMCost
from stellacode.costs.perimeter import PerimeterCost
from stellacode.surface.imports import get_cws


class AggregateCost:
    """Put together all the costs"""

    # def from_config(self, config):
    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)

        # Initialization of the different costs :
        self.EM = EMCost.from_config(config)
        self.S = get_cws(config)
        self.init_param = self.S.params

        self.lst_cost = [self.EM]
        if config["optimization_parameters"]["d_min"] == "True":
            self.dist = DistanceCost.from_config(config)
            self.lst_cost.append(self.dist)
        if config["optimization_parameters"]["perim"] == "True":
            self.perim = PerimeterCost.from_config(config)
            self.lst_cost.append(self.perim)
        if config["optimization_parameters"]["curvature"] == "True":
            self.curv = CurvatureCost.from_config(config)
            self.lst_cost.append(self.curv)

    def cost(self, **kwargs):
        tic = time()
        self.S.update_params(**kwargs)
        print("Surface", time() - tic)
        cost = 0.0
        for elt in self.lst_cost:
            tic = time()
            new_cost, _ = elt.cost(self.S)
            cost += new_cost
            print(elt.__class__.__name__, time() - tic)

        return cost
