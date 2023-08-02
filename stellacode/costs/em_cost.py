from typing import Optional

import pandas as pd
from jax import Array
from jax.typing import ArrayLike

import stellacode.tools as tools
from stellacode.tools import biot_et_savart, biot_et_savart_op
import stellacode.tools.bnorm as bnorm
from stellacode import mu_0_fac, np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.surface import AbstractSurface, IntegrationParams, FourierSurface
from stellacode.surface.imports import get_cws, get_plasma_surface
from stellacode.definitions import PlasmaConfig
from stellacode.tools.vmec import VMECIO
from .utils import merge_dataclasses

from pydantic import BaseModel, Extra


class BiotSavartOperator(BaseModel):
    """Interface for any cost"""

    class Config:
        arbitrary_types_allowed = True
        # extra = Extra.allow  # allow extra fields

    bs: ArrayLike
    use_mu_0_factor: bool = False

    def get_b_field(self, phi_mn):
        if len(self.bs.shape) == 4:
            b_field = np.einsum("hpqd,h", self.bs, phi_mn)
        else:
            b_field = np.einsum("hpq,h", self.bs, phi_mn)
        if not self.use_mu_0_factor:
            b_field *= mu_0_fac
        return b_field


class RegCoilSolver(BaseModel):
    """Interface for any cost"""

    class Config:
        arbitrary_types_allowed = True
        # extra = Extra.allow  # allow extra fields

    current_cov: ArrayLike
    matrix: ArrayLike
    matrix_reg: ArrayLike
    rhs: ArrayLike
    rhs_reg: ArrayLike
    net_currents: ArrayLike
    use_mu_0_factor: bool = False

    def solve_lambda(self, lamb: float = 0.0):
        if not self.use_mu_0_factor:
            lamb /= mu_0_fac**2
        if self.rhs_reg is not None:
            rhs = self.rhs - lamb * self.rhs_reg
        else:
            rhs = self.rhs

        phi_mn = np.linalg.solve(self.matrix + lamb * self.matrix_reg, rhs)

        if self.net_currents is not None:
            return np.concatenate((self.net_currents, phi_mn))
        else:
            return phi_mn


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
    net_currents: Optional[ArrayLike]
    Sp: AbstractSurface
    bnorm: ArrayLike = 0.0
    use_mu_0_factor: bool = False
    inverse_qj: bool = False
    slow_metrics: bool = True

    @classmethod
    def from_config(cls, config, Sp=None, use_mu_0_factor=True):
        curpol = float(config["other"]["curpol"])
        num_tor_symmetry = int(config["geometry"]["Np"])
        if Sp is None:
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

        return cls(
            lamb=float(config["other"]["lamb"]),
            net_currents=net_currents,
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
        )

    @classmethod
    def from_plasma_config(
        cls,
        plasma_config: PlasmaConfig,
        integration_par: IntegrationParams,
        Sp=None,
        use_mu_0_factor: bool = True,
        lamb: float = 0.1,
    ):
        if Sp is None:
            Sp = FourierSurface.from_file(plasma_config.path_plasma, integration_par=integration_par)

        vmec = VMECIO.from_grid(plasma_config.path_plasma)

        num_tor_symmetry = vmec.nfp

        net_currents = np.array(
            [
                vmec.net_poloidal_current / num_tor_symmetry,
                0.0,
            ]
        )

        # Checking the computation of net_poloidal_current
        # np.sum(vmec.b_cylindrical[-1] * vmec.grad_rphiz[-1][..., 1], axis=-1).sum(1) / mu_0*(1/(2*np.pi*5*48))
        # b_cart= vmec.b_cartesian[-1]
        # from scipy.constants import mu_0
        # np.sum(b_cart[:,:48]*Sp.jac_xyz[...,1], axis=-1).sum(1)/mu_0/48*num_tor_symmetry
        # np.linalg.norm(vmec.b_cartesian[-1], axis=-1)

        if plasma_config.path_bnorm is not None:
            bnorm_ = -vmec.scale_bnorm(bnorm.get_bnorm(plasma_config.path_bnorm, Sp))
            if not use_mu_0_factor:
                bnorm_ /= mu_0_fac
                # net_currents /= mu_0_fac
        else:
            bnorm_ = 0.0

        return cls(
            lamb=lamb,
            net_currents=net_currents,
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
        )

    def cost(self, S, results: Results = Results()):
        if "phi_mn" not in S.get_trainable_params().keys():
            solver, bs = self.get_regcoil_solver(S=S)
            phi_mn = solver.solve_lambda(lamb=self.lamb)
            bnorm_pred = bs.get_b_field(phi_mn)
        else:
            phi_mn = S.current.get_phi_mn()

            # old way, much more memory intensive
            # bs = self.get_bs_operator(S=S)
            # bnorm_pred = bs.get_b_field(phi_mn)
            bnorm_pred = S.get_b_field(xyz_plasma=self.Sp.xyz, plasma_normal=self.Sp.normal_unit)

            solver = None

        cost, metrics, results_ = self.get_results(
            bnorm_pred=bnorm_pred, solver=solver, phi_mn=phi_mn, S=S, lamb=self.lamb
        )
        return cost, metrics, merge_dataclasses(results, results_)

    def get_bs_operator(self, S, normal_b_field: bool = True):
        if normal_b_field:
            kwargs = dict(plasma_normal=self.Sp.normal_unit)
        else:
            kwargs = {}
        BS = S.get_b_field_op(xyz_plasma=self.Sp.xyz, **kwargs, scale_by_mu0=self.use_mu_0_factor)

        return BiotSavartOperator(bs=BS, use_mu_0_factor=self.use_mu_0_factor)

    def get_regcoil_solver(self, S):
        bs = self.get_bs_operator(S)
        BS = bs.bs
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

        return (
            RegCoilSolver(
                current_cov=Qj,
                matrix=matrix,
                matrix_reg=matrix_reg,
                rhs=RHS,
                rhs_reg=RHS_lamb,
                net_currents=self.net_currents,
                use_mu_0_factor=self.use_mu_0_factor,
            ),
            bs,
        )

    def get_results(self, bnorm_pred, phi_mn, S, solver=None, lamb: float = 0.0):
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac
        lamb /= fac**2

        metrics = {}
        results = Results(bnorm_plasma_surface=bnorm_pred, phi_mn_wnet_cur=phi_mn)
        B_err = bnorm_pred - self.bnorm * fac
        metrics["err_max_B"] = np.max(np.abs(B_err))

        metrics["cost_B"] = self.Sp.integrate(B_err**2)
        if solver is not None:
            metrics["cost_J"] = S.num_tor_symmetry * np.einsum("i,ij,j->", phi_mn, solver.current_cov, phi_mn)
            metrics["cost"] = metrics["cost_B"] + lamb * metrics["cost_J"]
        else:
            metrics["cost"] = metrics["cost_B"]

        if self.slow_metrics:
            j_3D = S.get_j_3D(phi_mn)
            metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2))
            results.j_3d = j_3D

            j_2D = S.get_j_surface(phi_mn)
            metrics["min_j_pol"] = j_2D[:, :, 0].min()
            results.j_s = j_2D

        return metrics["cost"], metrics, results

    def cost_multiple_lambdas(self, S, lambdas):
        solver, bs = self.get_regcoil_solver(S=S)
        results = {}
        for lamb in lambdas:
            phi_mn = solver.solve_lambda(lamb=lamb)
            bnorm_pred = bs.get_b_field(phi_mn)
            results[float(lamb)] = to_float(
                self.get_results(bnorm_pred=bnorm_pred, solver=solver, phi_mn=phi_mn, S=S, lamb=lamb)[1]
            )

        return pd.DataFrame(results).T

    def get_current_weights(self, S):
        solver, bs = self.get_regcoil_solver(S=S)
        return solver.solve_lambda(lamb=self.lamb)

    def get_b_field(self, coil_surf):
        solver = self.get_regcoil_solver(coil_surf)[0]
        phi_mn = solver.solve_lambda(self.lamb)
        bs = self.get_bs_operator(coil_surf, normal_b_field=False)
        return bs.get_b_field(phi_mn)


def to_float(dict_):
    return {k: float(v) for k, v in dict_.items()}


def get_b_field_err(em_cost, coil_surf, err: str = "rmse_n"):
    b_field = em_cost.get_b_field(coil_surf)
    vmec = VMECIO.from_grid(
        em_cost.Sp.file_path,
        ntheta=em_cost.Sp.integration_par.num_points_u,
        nzeta=em_cost.Sp.integration_par.num_points_v * em_cost.Sp.num_tor_symmetry,
        surface_label=-1,
    )
    b_field_gt = np.transpose(vmec.b_cartesian[-1, :, : em_cost.Sp.integration_par.num_points_v], (2, 0, 1))
    if err == "rmse_n":
        return np.sqrt(em_cost.Sp.integrate((b_field - b_field_gt) ** 2) / em_cost.Sp.area)
    elif err == "max":
        return np.max(np.linalg.norm(b_field_gt - b_field, axis=0))
