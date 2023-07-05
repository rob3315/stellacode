from typing import Optional

import pandas as pd
from jax import Array
from jax.typing import ArrayLike

import stellacode.tools as tools
import stellacode.tools.bnorm as bnorm
from stellacode import mu_0_fac, np
from stellacode.costs.abstract_cost import AbstractCost
from stellacode.surface.abstract_surface import AbstractSurface
from stellacode.surface.imports import get_cws_grid, get_plasma_surface


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
    rot_tensor: ArrayLike
    matrixd_phi: ArrayLike
    use_mu_0_factor: bool = False

    @classmethod
    def from_config(cls, config, use_mu_0_factor=True):
        mpol_coil = int(config["geometry"]["mpol_coil"])
        ntor_coil = int(config["geometry"]["ntor_coil"])
        curpol = float(config["other"]["curpol"])
        num_tor_symmetry = int(config["geometry"]["Np"])
        rot_tensor = tools.get_rot_tensor(num_tor_symmetry)
        phisize = (mpol_coil, ntor_coil)
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
            rot_tensor=rot_tensor,
            matrixd_phi=tools.get_matrix_dPhi(phisize, get_cws_grid(config)),
            use_mu_0_factor=use_mu_0_factor,
        )

    def cost(self, S):
        BS = self.get_BS_norm(S)
        j_S, Qj = self.get_current(BS=BS, S=S, lamb=self.lamb)
        return self.get_results(BS=BS, j_S=j_S, S=S, Qj=Qj, lamb=self.lamb)

    def get_BS_norm(self, S):
        Sp = self.Sp
        r_plasma = Sp.P

        # rotate and duplicate coil
        # TODO: should be exported to the surface class
        # r_coil = np.reshape(
        #     np.einsum("opq,ijq->oijp", self.rot_tensor, S.P), (-1, S.nbpts[1], 3)
        # )
        # surface_current = np.concatenate(
        #     [self.matrixd_phi] * self.rot_tensor.shape[0], 1
        # )
        self.matrixd_phi = S.get_curent_potential_op()
        # jac_r_plasma = np.reshape(
        #     np.einsum("sba,taij->sijbt", self.rot_tensor, S.dpsi),
        #     (-1, S.nbpts[1], 3, 2),
        # )
        r_coil = S.P
        jac_r_plasma = S.dpsi

        BS = biot_et_savart(r_plasma, r_coil, self.matrixd_phi, jac_r_plasma) / S.npts

        BS = np.einsum("tpqd,dpq->tpq", BS, Sp.n)
        if self.use_mu_0_factor:
            BS *= mu_0_fac
        return BS

    def get_current(self, BS, S, lamb: float = 0.0):
        Sp = self.Sp

        # Now we have to compute the best current components.
        # This is a technical part, one should read the paper :
        # "Optimal shape of stellarators for magnetic confinement fusion"
        # in order to understand what's going on.

        # need to rotate dpsi
        Qj = tools.compute_Qj(self.matrixd_phi, S.dpsi, S.dS)

        if self.net_currents is not None:
            BS_R = BS[2:]
            Qj_inv_R = np.linalg.inv(Qj[2:, 2:])
            bnorm_ = self.bnorm - np.einsum("tpq,t", BS[:2], self.net_currents)
        else:
            BS_R = BS
            Qj_inv_R = np.linalg.inv(Qj)
            bnorm_ = self.bnorm

        BS_dagger = np.einsum("ut,tij,ij->uij", Qj_inv_R, BS_R, Sp.dS / Sp.npts)

        RHS = np.einsum("hpq,pq->h", BS_dagger, bnorm_)

        if self.net_currents is not None:
            RHS_lamb = Qj_inv_R @ Qj[2:, :2] @ self.net_currents
        else:
            RHS_lamb = None

        j_S = self.solve_lambda(
            BS_R=BS_R,
            BS_dagger=BS_dagger,
            RHS=RHS,
            RHS_lamb=RHS_lamb,
            lamb=lamb,
        )

        return j_S, Qj

    def get_results(self, BS, j_S, S, Qj, lamb: float = 0.0):
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac
        lamb /= fac**2

        j_3D = self.get_j_3D(j_S, S)
        metrics = {}
        bnorm_pred = np.einsum("hpq,h", BS, j_S)
        B_err = bnorm_pred - self.bnorm
        metrics["err_max_B"] = np.max(np.abs(B_err)) * fac
        metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2)) * fac
        metrics["cost_B"] = (
            self.num_tor_symmetry * np.sum(B_err**2 * self.Sp.dS) / self.Sp.npts * fac**2
        )

        metrics["cost_J"] = self.num_tor_symmetry * np.einsum("i,ij,j->", j_S, Qj, j_S)
        metrics["cost"] = metrics["cost_B"] + lamb * metrics["cost_J"]

        return metrics["cost"], metrics

    def solve_lambda(self, BS_R, BS_dagger, RHS, RHS_lamb=None, lamb: float = 0.0):
        if not self.use_mu_0_factor:
            lamb /= mu_0_fac**2
        inside_M_lambda_R = lamb * np.eye(BS_R.shape[0]) + np.einsum(
            "tpq,upq->tu", BS_dagger, BS_R
        )
        M_lambda_R = np.linalg.inv(inside_M_lambda_R)

        if self.net_currents is not None:
            RHS = RHS - lamb * RHS_lamb

        j_S_R = M_lambda_R @ RHS

        if self.net_currents is not None:
            j_S = np.concatenate((self.net_currents, j_S_R))
        else:
            j_S = j_S_R

        return j_S

    def get_j_3D(self, j_S, S):
        # j_S is a vector containing the components of the best scalar current potential.
        # The real surface current is given by :
        return np.einsum(
            "oijk,ijdk,ij,o->ijd",
            self.matrixd_phi,
            S.dpsi,
            1 / S.dS,
            j_S,
            optimize=True,
        )

    def cost_multiple_lambdas(self, S, lambdas):
        BS = self.get_BS_norm(S)
        results = {}
        for lamb in lambdas:
            j_S, Qj = self.get_current(BS=BS, S=S, lamb=lamb)
            results[float(lamb)] = to_float(
                self.get_results(BS=BS, j_S=j_S, S=S, Qj=Qj, lamb=lamb)[1]
            )

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
