from typing import Optional

import pandas as pd
from jax import Array
from jax.typing import ArrayLike

import stellacode.tools as tools
import stellacode.tools.bnorm as bnorm
from stellacode import mu_0_fac, np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.surface.abstract_surface import AbstractSurface
from stellacode.surface.imports import get_cws, get_plasma_surface


class EMCost(AbstractCost):
    """Main cost coming from the inverse problem

    Args:
        * Regularization parameter
        * Number of field periods
        * Number of poloidal points on the cws
        * Number of toroidal points on the cws
        * Amount of current flowig poloidally
        * Amount of current flowig toroidally (usually 0)
    """

    lamb: float
    num_tor_symmetry: int
    net_currents: Optional[ArrayLike]
    Sp: AbstractSurface
    bnorm: ArrayLike
    use_mu_0_factor: bool = False
    inverse_qj: bool = False

    @classmethod
    def from_config(cls, config, use_mu_0_factor=True):
        curpol = float(config["other"]["curpol"])
        num_tor_symmetry = int(config["geometry"]["Np"])
        Sp = get_plasma_surface(config)
        bnorm_ = -curpol * bnorm.get_bnorm(str(config["other"]["path_bnorm"]), Sp)
        net_currents = np.array(
            [
                float(config["other"]["net_poloidal_current_Amperes"]) / num_tor_symmetry,
                float(config["other"]["net_toroidal_current_Amperes"]),
            ]
        )
        if not use_mu_0_factor:
            bnorm_ /= mu_0_fac
            # net_currents /= mu_0_fac

        return cls(
            lamb=float(config["other"]["lamb"]),
            num_tor_symmetry=num_tor_symmetry,
            net_currents=net_currents,
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
        )

    def cost(self, S):
        BS = self.get_BS_norm(S)
        j_S, Qj = self.get_current(BS=BS, S=S, lamb=self.lamb)
        return self.get_results(BS=BS, j_S=j_S, S=S, Qj=Qj, lamb=self.lamb)

    def get_current_result(self, S):
        BS = self.get_BS_norm(S)
        return self.get_current(BS=BS, S=S, lamb=self.lamb)[0]

    def get_BS_norm(self, S):
        Sp = self.Sp
        r_plasma = Sp.xyz

        r_coil = S.xyz
        jac_xyz = S.jac_xyz

        BS = biot_et_savart(r_plasma, r_coil, S.current_op, jac_xyz) / S.npts

        BS = np.einsum("tpqd,dpq->tpq", BS, Sp.normal_unit)
        if self.use_mu_0_factor:
            BS *= mu_0_fac
        return BS

    def get_b_field(self, BS, j_S):
        return np.einsum("hpqd,h", BS, j_S)

    def get_current(self, BS, S, lamb: float = 0.0):
        Sp = self.Sp

        # Now we have to compute the best current components.
        # This is a technical part, one should read the paper :
        # "Optimal shape of stellarators for magnetic confinement fusion"
        # in order to understand what's going on.

        Qj = tools.compute_Qj(S.current_op, S.jac_xyz, S.ds)

        if self.net_currents is not None:
            BS_R = BS[2:]
            if self.inverse_qj:
                Qj_inv_R = np.linalg.inv(Qj[2:, 2:])
            bnorm_ = self.bnorm - np.einsum("tpq,t", BS[:2], self.net_currents)
        else:
            BS_R = BS
            if self.inverse_qj:
                Qj_inv_R = np.linalg.inv(Qj)
            bnorm_ = self.bnorm
        if self.inverse_qj:
            BS_dagger = np.einsum("ut,tij,ij->uij", Qj_inv_R, BS_R, Sp.ds / Sp.npts)
        else:
            BS_dagger = np.einsum("uij,ij->uij", BS_R, Sp.ds / Sp.npts)

        RHS = np.einsum("hpq,pq->h", BS_dagger, bnorm_)

        if self.net_currents is not None:
            if self.inverse_qj:
                RHS_lamb = Qj_inv_R @ Qj[2:, :2] @ self.net_currents
            else:
                RHS_lamb = Qj[2:, :2] @ self.net_currents
        else:
            RHS_lamb = None

        matrix = np.einsum("tpq,upq->tu", BS_dagger, BS_R)
        if self.inverse_qj:
            matrix_reg = np.eye(BS_R.shape[0])
        else:
            matrix_reg = Qj[2:, 2:]

        j_S = self.solve_lambda(
            matrix=matrix,
            matrix_reg=matrix_reg,
            rhs=RHS,
            rhs_reg=RHS_lamb,
            lamb=lamb,
        )

        return j_S, Qj

    def get_results(self, BS, j_S, S, Qj, lamb: float = 0.0):
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac
        lamb /= fac**2

        j_3D = S.get_j_3D(j_S)
        metrics = {}
        bnorm_pred = np.einsum("hpq,h", BS, j_S)
        B_err = bnorm_pred - self.bnorm
        metrics["err_max_B"] = np.max(np.abs(B_err)) * fac
        metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2)) * fac
        metrics["cost_B"] = self.num_tor_symmetry * np.sum(B_err**2 * self.Sp.ds) / self.Sp.npts * fac**2

        metrics["cost_J"] = self.num_tor_symmetry * np.einsum("i,ij,j->", j_S, Qj, j_S)
        metrics["cost"] = metrics["cost_B"] + lamb * metrics["cost_J"]

        return metrics["cost"], metrics

    def solve_lambda(self, matrix, matrix_reg, rhs, rhs_reg=None, lamb: float = 0.0):
        if not self.use_mu_0_factor:
            lamb /= mu_0_fac**2

        if self.net_currents is not None:
            rhs = rhs - lamb * rhs_reg

        j_S_R = np.linalg.solve(matrix + lamb * matrix_reg, rhs)

        if self.net_currents is not None:
            j_S = np.concatenate((self.net_currents, j_S_R))
        else:
            j_S = j_S_R

        return j_S

    def cost_multiple_lambdas(self, S, lambdas):
        BS = self.get_BS_norm(S)
        results = {}
        for lamb in lambdas:
            j_S, Qj = self.get_current(BS=BS, S=S, lamb=lamb)
            results[float(lamb)] = to_float(self.get_results(BS=BS, j_S=j_S, S=S, Qj=Qj, lamb=lamb)[1])

        return pd.DataFrame(results).T


def to_float(dict_):
    return {k: float(v) for k, v in dict_.items()}


def biot_et_savart(
    r_plasma: ArrayLike,
    r_coil: ArrayLike,
    surface_current: ArrayLike,
    jac_r_plasma: ArrayLike,
) -> Array:
    """
    Args:
     * r_plasma: dims = up x vp x 3
     * r_coil: dims = uc x vc x 3
     * surface_current: dims = uc x vc x 3 or current_basis x uc x vc x 3
     * jac_r_plasma: dims = uc x vc x 3 x 2
    """

    T = r_plasma[None, None, ...] - r_coil[:, :, None, None]
    K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

    sc_jac = np.einsum("tijh,ijah->ijat", surface_current, jac_r_plasma)
    B = np.einsum("ijpqa,ijbt, dab->tpqd", K, sc_jac, tools.eijk)

    return B
