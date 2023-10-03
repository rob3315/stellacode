from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import Constraint


class LaplaceForceCost(AbstractCost):
    """
    Constraint on the Laplace Force.
    
    Args:
        * nfp: number of field periods
        * constraint: compute the constraint cost on the laplace force
        * normalization: normalization factor before applying the constraint cost.
    """

    nfp: int
    constraint: Constraint = Constraint(limit=0.0, distance=1.0, weight=1.0, minimum=False)
    normalization: float = 1e6

    def cost(self, S, results: Results = Results()):
        lap_force = S.laplace_force(nfp=self.nfp)
        lap_force_norm = np.linalg.norm(lap_force, axis=-1)
        loss = self.constraint.barrier(lap_force_norm).mean()

        return loss, {"cost_laplace_force": loss}, results
