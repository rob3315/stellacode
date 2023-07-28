from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import inverse_barrier


class CurrentCtrCost(AbstractCost):
    """Penalization on the maximal current"""

    max_val: float
    distance: float = 1.0
    weight: float = 1.0
    normalization: float = 1e6

    def cost(self, S, results: Results = Results()):
        j_3d = results.j_3d / self.normalization
        assert j_3d is not None
        j_3d_norm = np.linalg.norm(j_3d, axis=-1)
        loss = inverse_barrier(
            val=-j_3d_norm,
            min_val=-self.max_val,
            distance=self.distance,
            weight=self.weight,
        )

        return loss.sum(), {}, results
