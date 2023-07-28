from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import inverse_barrier


class CurvatureCost(AbstractCost):
    """Non linear penalization on the curvature (upper bound)"""

    max_val: float = 1.
    distance: float = 0.1
    weight: float = 1.0

    @classmethod
    def from_config(cls, config, Sp=None):
        c0 = float(config["optimization_parameters"]["curvature_c0"])
        c1 = float(config["optimization_parameters"]["curvature_c1"])
        return cls(
            distance=c1 - c0,
            max_val=c1,
        )

    def cost(self, S, results: Results = Results()):
        pmax, pmin = S.principles[0], S.principles[1]
        f_pmax = inverse_barrier(val=-pmax, min_val=-self.max_val, distance=self.distance, weight=self.weight)
        f_pmin = inverse_barrier(val=-pmin, min_val=-self.max_val, distance=self.distance, weight=self.weight)
        # cost = S.integrate(f_pmax+f_pmin)
        cost = np.einsum("ij,ij->", f_pmax, S.ds / S.npts)
        cost += np.einsum("ij,ij->", f_pmin, S.ds / S.npts)
        aux_dic = {}
        aux_dic["max_curvature"] = max(np.max(pmax), np.max(-pmin))

        return cost, aux_dic, results


class NegTorCurvatureCost(AbstractCost):
    """Non linear penalization on the negative toroidal curvatures"""

    min_val: float = 0.0
    distance: float = 0.1
    weight: float = 1.0

    def cost(self, S, results: Results = Results()):
        # get normalized curvature along the poloidal dimension
        loss = inverse_barrier(
            self.get_toroidal_curvature(S),
            min_val=self.min_val,
            distance=self.distance,
            weight=self.weight,
        ).mean()

        return loss.sum(), {"min_v_curvature": S.hess_xyz[..., 1, 1].min()}, results

    def get_toroidal_curvature(self, S):
        # TODO: check these formulae
        curv = np.einsum("ijp, ijp ->ij", S.normal_unit, S.hess_xyz[..., 0, 0])
        norm_curv = curv / S.ds
        return norm_curv
