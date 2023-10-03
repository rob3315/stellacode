from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import Constraint, inverse_barrier
from pydantic import BaseModel, Extra
from stellacode.tools.utils import cross


class LaplaceForceCost(AbstractCost):
    """Constraint on the Laplace Force."""

    num_tor_symmetry: int
    constraint: Constraint = Constraint(limit=0.0, distance=1.0, weight=1.0, minimum=False)
    normalization: float = 1e6


    def cost(self, S, results: Results = Results()):
        lap_force = S.laplace_force(num_tor_symmetry=self.num_tor_symmetry)
        lap_force_norm = np.linalg.norm(lap_force, axis=-1)
        loss = self.constraint.barrier(lap_force_norm).mean()

        return loss, {"cost_laplace_force": loss}, results
