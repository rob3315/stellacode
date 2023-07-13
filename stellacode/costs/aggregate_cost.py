import configparser
from time import time

from stellacode.costs.area import AreaCost
from stellacode.costs.curvature import CurvatureCost
from stellacode.costs.distance import DistanceCost
from stellacode.costs.em_cost import EMCost
from stellacode.surface.imports import get_cws, get_plasma_surface
from .abstract_cost import AbstractCost


class AggregateCost(AbstractCost):
    """Put together all the costs"""

    costs: list

    @classmethod
    def from_config(cls, config, Sp=None):
        if Sp is None:
            Sp = get_plasma_surface(config)
        costs = [EMCost.from_config(config, Sp=Sp)]
        if config["optimization_parameters"]["d_min"] == "True":
            costs.append(DistanceCost.from_config(config, Sp=Sp))
        if config["optimization_parameters"]["perim"] == "True":
            costs.append(AreaCost.from_config(config, Sp=Sp))
        if config["optimization_parameters"]["curvature"] == "True":
            costs.append(CurvatureCost.from_config(config, Sp=Sp))
        return cls(costs=costs)

    def cost(self, S):
        cost = 0.0
        metrics_d = {}
        for elt in self.costs:
            tic = time()
            new_cost, metrics = elt.cost(S)
            metrics_d = {**metrics_d, **metrics}
            cost += new_cost
            print(elt.__class__.__name__, time() - tic)

        return cost, metrics_d
