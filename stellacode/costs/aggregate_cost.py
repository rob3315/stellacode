import configparser
from time import time

from stellacode.costs.area import AreaCost
from stellacode.costs.curvature import CurvatureCost
from stellacode.costs.distance import DistanceCost
from stellacode.costs.em_cost import EMCost
from stellacode.surface.imports import get_cws, get_plasma_surface

from .abstract_cost import AbstractCost, Results
from .utils import merge_dataclasses


class AggregateCost(AbstractCost):
    """Sum a list of costs"""

    costs: list

    @classmethod
    def from_config(cls, config, Sp=None):
        if Sp is None:
            Sp = get_plasma_surface(config)()
        costs = [EMCost.from_config(config, Sp=Sp)]
        if config["optimization_parameters"]["d_min"] == "True":
            costs.append(DistanceCost.from_config(config, Sp=Sp))
        if config["optimization_parameters"]["perim"] == "True":
            costs.append(AreaCost.from_config(config, Sp=Sp))
        if config["optimization_parameters"]["curvature"] == "True":
            costs.append(CurvatureCost.from_config(config, Sp=Sp))
        return cls(costs=costs)

    def cost(self, S, results: Results = Results()):
        cost = 0.0
        metrics_d: dict = {}
        for elt in self.costs:
            new_cost, metrics, results_ = elt.cost(S, results=results)
            metrics_d = {**metrics_d, **metrics}
            merge_dataclasses(results, results_)
            cost += new_cost

        metrics_d["total_cost"] = cost

        return cost, metrics_d, results
