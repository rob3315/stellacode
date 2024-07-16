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
        """
        Create an instance of AggregateCost from a configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
            Sp (Surface, optional): The plasma surface. If None, it is created from the configuration.

        Returns:
            AggregateCost: An instance of AggregateCost.
        """
        # If Sp is None, create a plasma surface from the configuration
        if Sp is None:
            Sp = get_plasma_surface(config)()

        # Initialize the list of costs
        costs = [EMCost.from_config(config, Sp=Sp)]

        # If d_min is enabled, add a DistanceCost to the list of costs
        if config["optimization_parameters"]["d_min"] == "True":
            costs.append(DistanceCost.from_config(config, Sp=Sp))

        # If perim is enabled, add an AreaCost to the list of costs
        if config["optimization_parameters"]["perim"] == "True":
            costs.append(AreaCost.from_config(config, Sp=Sp))

        # If curvature is enabled, add a CurvatureCost to the list of costs
        if config["optimization_parameters"]["curvature"] == "True":
            costs.append(CurvatureCost.from_config(config, Sp=Sp))

        # Create and return an instance of AggregateCost with the extracted parameters
        return cls(costs=costs)

    def cost(self, S, results: Results = Results()):
        """
        Calculate the total cost of a given CWS for a given current distribution.
        This is done by summing individual costs and update metrics.

        Args:
            S: The CWS Surface.
            results (Results, optional): A given set of results. Defaults to Results().

        Returns:
            Tuple[float, Dict[str, float], Results, Surface]: The total cost, metrics, updated results, and Surface object.
        """
        # Total cost value
        cost = 0.0
        # Metrics dictionnary
        metrics_d: dict = {}

        # Calculate individual costs and metrics in a given order
        for elt in self.costs:
            # Compute the cost, metrics and results for a given CWS + current configuration
            new_cost, metrics, results_, S = elt.cost(S, results=results)
            # Update the metrics dictionnary with the new metrics info
            metrics_d = {**metrics_d, **metrics}
            # Update the results
            merge_dataclasses(results, results_)
            # Update the cost
            cost += new_cost

        # Update the metrics with the total cost
        metrics_d["total_cost"] = cost

        return cost, metrics_d, results, S
