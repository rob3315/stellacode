from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import Constraint, inverse_barrier
from stellacode.tools.hts_critical_current import HTSCriticalCurrent


class CurrentCtrCost(AbstractCost):
    """Penalization on the maximal current"""

    constraint: Constraint = Constraint(limit=0.0, distance=1.0, weight=1.0, minimum=False)
    normalization: float = 1e6

    def cost(self, S, results: Results = Results()):
        j_3d = results.j_3d / self.normalization
        assert j_3d is not None
        j_3d_norm = np.linalg.norm(j_3d, axis=-1)
        loss = self.constraint.barrier(j_3d_norm).mean()

        return loss, {"cost_j_ctr": loss}, results, S


class PoloidalCurrentCost(AbstractCost):
    """
    Penalization on the maximal or minimal poloidal current.

    It is a naive solution to avoid whirlpools of currents.
    It may be overconstraining.
    """

    constraint: Constraint = Constraint(limit=0.0, distance=1.0, weight=1.0, minimum=False)
    normalization: float = 1e6

    def cost(self, S, results: Results = Results()):
        ju = S.j_surface[:, :, 0]

        loss = self.constraint.barrier(ju).mean()

        return loss, {"cost_ju_ctr": loss, "max_ju": np.max(ju), "min_ju": np.min(ju)}, results, S


class CriticalCurrentCtr(AbstractCost):
    """
    Penalization on the critical current

    The current implementation is naive (the total current limit is n times the
    HTS tape critical current).
    A better implementation would be as follows:
        * The current inside each layer is Ji and the total current is J
        * The magnetic field at position i is: Bi=B0*sum k<i Jk/J

    We have the constraints:
        * Ji<Jc
        * sum Ji=J

    It may be possible to solve once and for all this set of equations:
    We want to minimize the risk of going beyond the critical current: max_i Ji(Bi)/Jc(Bi)
    with the constraints that: sum Ji=J

    Which can be translated into the following problem (with one lagrangian multiplier):
        * grad max(Ji(Bi)/Jc(Bi))-lambda * (J-sum Ji(Bi))=0
        * J-sum Ji(Bi)=0

    """

    critical_current_tape: HTSCriticalCurrent = HTSCriticalCurrent()
    constraint: Constraint = Constraint(limit=0.0, distance=1.0, weight=1.0, minimum=False)
    normalization: float = 1e6
    temperature: float = 10.0
    num_hts_turn: float = 100

    def cost(self, S, results: Results = Results()):
        j_3d = results.j_3d
        b_field = S.get_b_field(xyz_plasma=S.xyz)
        theta = np.arctan2(j_3d / np.norm(j_3d, axis=-1), b_field / np.norm(b_field, axis=-1))
        jc = self.critical_current_tape(b_field, theta, temperature=self.temperature) * self.num_hts_turn

        j_3d_norm = np.linalg.norm(j_3d, axis=-1)

        loss = self.constraint.barrier(j_3d_norm - jc).mean()

        return loss, {"cost_j_ctr": loss}, results, S
