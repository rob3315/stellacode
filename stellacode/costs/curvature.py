from stellacode import np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.costs.utils import Constraint, inverse_barrier


class CurvatureCost(AbstractCost):
    """Non linear penalization on the curvature (upper bound)"""

    constraint: Constraint = Constraint(limit=1, distance=0.1, weight=1.0, minimum=False)


    @classmethod
    def from_config(cls, config, Sp=None):
        c0 = float(config["optimization_parameters"]["curvature_c0"])
        c1 = float(config["optimization_parameters"]["curvature_c1"])
        return cls(constraint=Constraint(limit=c1, distance=c1 - c0, minimum=False))

    def cost(self, S, results: Results = Results()):
        pmax, pmin = S.principle_max, S.principle_min
        f_pmax = self.constraint.barrier(pmax)
        f_pmin = self.constraint.barrier(pmin)
        cost = np.einsum("ij,ij->", f_pmax, S.ds / S.npts)
        cost += np.einsum("ij,ij->", f_pmin, S.ds / S.npts)
        aux_dic = {}
        aux_dic["max_curvature"] = max(np.max(pmax), np.max(-pmin))

        return cost, aux_dic, results


class NegTorCurvatureCost(AbstractCost):
    """Non linear penalization on the negative toroidal curvatures"""

    constraint: Constraint = Constraint(limit=0.0, distance=0.1, weight=1.0, minimum=True)


    def cost(self, S, results: Results = Results()):
        # get normalized curvature along the poloidal dimension
        curvature = self.get_toroidal_curvature(S)
        loss = self.constraint.barrier(curvature).sum()

        return loss, {"min_v_curvature": curvature.min(), "cost_neg_pol_curv": loss}, results

    def get_toroidal_curvature(self, S):
        # TODO: check these formulae
        curv = np.einsum("ijp, ijp ->ij", S.normal_unit, S.hess_xyz[..., 0, 0])
        norm_curv = curv / S.ds
        return norm_curv
