import typing as tp

import pandas as pd
from jax import Array
from jax.typing import ArrayLike
from pydantic import BaseModel

from stellacode import mu_0_fac, np
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.definitions import PlasmaConfig
from stellacode.surface import FourierSurfaceFactory, IntegrationParams, Surface
from stellacode.surface.imports import get_plasma_surface
from stellacode.tools.bnorm import get_bnorm
from stellacode.tools.vmec import VMECIO

from .utils import merge_dataclasses


class BiotSavartOperator(BaseModel):
    """
    Represent a biot et savart operator

    Args:
        * bs: biot est savart operator tensor
        * use_mu_0_factor: scale the operator tensor by mu_0 factor.
    """

    class Config:
        arbitrary_types_allowed = True

    bs: Array
    use_mu_0_factor: bool = False

    def get_b_field(self, phi_mn):
        dims = "pqabcd"[: len(self.bs.shape) - 1]
        b_field = np.einsum(f"h{dims},h->{dims}", self.bs, phi_mn)
        if not self.use_mu_0_factor:
            b_field *= mu_0_fac
        return b_field


class RegCoilSolver(BaseModel):
    """Solve the regcoil problem:

    (M + \lambda * Mr)\Phi_mn=rhs+\lambda * rhs_reg

    Args:
        * current_basis_dot_prod: scalar product matrix of the current basis functions
        * matrix: M
        * matrix_reg: Mr
        * rhs: rhs
        * rhs_reg: rhs_reg
        * net_currents: net poloidal and toroidal currents
        * biot_et_savart_op: biot et savard operator used in building the problem
        * use_mu_0_factor: divide inputs by mu_0_fac and multply outputs by mu_0_fac
    """

    class Config:
        arbitrary_types_allowed = True

    current_basis_dot_prod: ArrayLike
    matrix: ArrayLike
    matrix_reg: ArrayLike
    rhs: ArrayLike
    rhs_reg: ArrayLike
    net_currents: ArrayLike
    biot_et_savart_op: BiotSavartOperator
    use_mu_0_factor: bool = False

    def solve_lambda(self, lamb: float = 0.0):
        """
        Solve regcoil for a given lambda value
        """
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

    @classmethod
    def from_surfaces(
        cls,
        S,
        Sp,
        bnorm: tp.Optional[ArrayLike],
        normal_b_field: bool = True,
        use_mu_0_factor: bool = False,
        fit_b_3d: bool = False,
    ):
        if normal_b_field:
            kwargs = dict(plasma_normal=Sp.normal_unit)
        else:
            kwargs = {}
        BS = S.get_b_field_op(xyz_plasma=Sp.xyz, **kwargs, scale_by_mu0=use_mu_0_factor)

        bs = BiotSavartOperator(bs=BS, use_mu_0_factor=use_mu_0_factor)
        BS = bs.bs

        # Now we have to compute the best current components.
        # This is a technical part, one should read the paper :
        # "Optimal shape of stellarators for magnetic confinement fusion"
        # in order to understand what's going on.

        current_basis_dot_prod = S.get_current_basis_dot_prod()
        if fit_b_3d:
            add_dim = "a"
        else:
            add_dim = ""

        if S.net_currents is not None:
            BS_R = BS[2:]
            bnorm_ = bnorm - np.einsum(f"tpq{add_dim},t->pq{add_dim}", BS[:2], S.net_currents)
        else:
            BS_R = BS
            bnorm_ = bnorm
        BS_dagger = np.einsum(f"uij{add_dim},ij->uij{add_dim}", BS_R, Sp.ds / Sp.npts)

        rhs = np.einsum(f"hpq{add_dim},pq{add_dim}->h", BS_dagger, bnorm_)

        if S.net_currents is not None:
            rhs_reg = current_basis_dot_prod[2:, :2] @ S.net_currents
        else:
            rhs_reg = None

        matrix = np.einsum(f"tpq{add_dim},upq{add_dim}->tu", BS_dagger, BS_R)
        matrix_reg = current_basis_dot_prod[2:, 2:]

        return cls(
            current_basis_dot_prod=current_basis_dot_prod,
            matrix=matrix,
            matrix_reg=matrix_reg,
            rhs=rhs,
            rhs_reg=rhs_reg,
            net_currents=S.net_currents,
            use_mu_0_factor=use_mu_0_factor,
            biot_et_savart_op=bs,
        )


class MSEBField(AbstractCost):
    """
    Mean Squared error on the plasma magnetic field

    Args:
        * Sp: plasma surface
        * bnorm: normal magnetic field on the plasma surface
        * use_mu_0_factor: Multiply by the mu_0 factor.
        * slow_metrics: compute metrics that take time.
        * train_currents: use current parameters to compute the currents
        * fit_b_3d: the 3D field is regressed instead of the normal magnetic field
    """

    Sp: Surface
    bnorm: ArrayLike = 0.0
    use_mu_0_factor: bool = False
    slow_metrics: bool = True
    train_currents: bool = False
    fit_b_3d: bool = False

    @classmethod
    def from_plasma_config(
        cls,
        plasma_config: PlasmaConfig,
        integration_par: IntegrationParams,
        Sp=None,
        use_mu_0_factor: bool = True,
        fit_b_3d: bool = False,
        surface_label=-1,
    ):
        if Sp is None:
            Sp = FourierSurfaceFactory.from_file(
                plasma_config.path_plasma,
                integration_par=integration_par,
            )()

        vmec = VMECIO.from_grid(
            plasma_config.path_plasma,
            ntheta=integration_par.num_points_u,
            nzeta=integration_par.num_points_v,
            surface_label=surface_label,
        )

        if plasma_config.path_bnorm is not None:
            bnorm_ = -vmec.scale_bnorm(get_bnorm(plasma_config.path_bnorm, Sp))
            if not use_mu_0_factor:
                bnorm_ /= mu_0_fac
        else:
            bnorm_ = 0.0

        if fit_b_3d:
            bnorm_ = vmec.b_cartesian[-1]

        return cls(
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
            fit_b_3d=fit_b_3d,
        )

    def cost(self, S, results: Results = Results()):
        if not self.fit_b_3d:
            plasma_normal = self.Sp.normal_unit
        else:
            plasma_normal = None

        bnorm_pred = S.get_b_field(xyz_plasma=self.Sp.xyz, plasma_normal=plasma_normal)

        cost, metrics, results_ = self.get_results(bnorm_pred=bnorm_pred, S=S)
        return cost, metrics, merge_dataclasses(results, results_)

    def get_results(self, bnorm_pred, S):
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac

        metrics = {}
        results = Results(bnorm_plasma_surface=bnorm_pred)
        b_err = (bnorm_pred - self.bnorm * fac) ** 2
        metrics["err_max_B"] = np.max(np.abs(b_err))
        if self.fit_b_3d:
            b_err = np.sum(b_err, axis=-1)
        metrics["cost_B"] = self.Sp.integrate(b_err)
        metrics["cost"] = metrics["cost_B"]

        if self.slow_metrics:
            j_3D = S.j_3d
            metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2))
            results.j_3d = j_3D

        return metrics["cost"], metrics, results


class EMCost(AbstractCost):
    """
    Cost of the optimal current inverse problem.

    Args:
        * lamb: Regularization parameter.
        * Sp: Plasma surface
        * bnorm: normal magnetic field pointing inside of the surface.
        * use_mu_0_factor: Multiply by the mu_0 factor.
        * slow_metrics: compute metrics that take time.
        * train_currents: use current parameters to compute the currents
        * fit_b_3d: the 3D field is regressed instead of the normal magnetic field
    """

    lamb: float
    Sp: Surface
    bnorm: ArrayLike = 0.0
    use_mu_0_factor: bool = False
    slow_metrics: bool = True
    train_currents: bool = False
    fit_b_3d: bool = False

    @classmethod
    def from_config(cls, config, Sp=None, use_mu_0_factor=True):
        curpol = float(config["other"]["curpol"])
        if Sp is None:
            Sp = get_plasma_surface(config)()

        bnorm_ = -curpol * get_bnorm(str(config["other"]["path_bnorm"]), Sp)

        if not use_mu_0_factor:
            bnorm_ /= mu_0_fac

        return cls(
            lamb=float(config["other"]["lamb"]) / int(config["geometry"]["Np"]),
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
        train_currents: bool = False,
        fit_b_3d: bool = False,
        surface_label=-1,
    ):
        if Sp is None:
            Sp = FourierSurfaceFactory.from_file(plasma_config.path_plasma, integration_par=integration_par)()

        vmec = VMECIO.from_grid(
            plasma_config.path_plasma,
            ntheta=integration_par.num_points_u,
            nzeta=integration_par.num_points_v,
            surface_label=surface_label,
        )

        if plasma_config.path_bnorm is not None:
            bnorm_ = -vmec.scale_bnorm(get_bnorm(plasma_config.path_bnorm, Sp))
            if not use_mu_0_factor:
                bnorm_ /= mu_0_fac
        else:
            bnorm_ = 0.0

        if fit_b_3d:
            bnorm_ = vmec.b_cartesian[-1]

        return cls(
            lamb=lamb,
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
            train_currents=train_currents,
            fit_b_3d=fit_b_3d,
        )

    def cost(self, S, results: Results = Results()):
        if not self.train_currents:
            solver = self.get_regcoil_solver(S=S)
            phi_mn = solver.solve_lambda(lamb=self.lamb)
            bnorm_pred = solver.biot_et_savart_op.get_b_field(phi_mn)
        else:
            phi_mn = None

            # old way, much more memory intensive
            # bs = self.get_bs_operator(S=S)
            # bnorm_pred = bs.get_b_field(phi_mn)
            if not self.fit_b_3d:
                plasma_normal = self.Sp.normal_unit
            else:
                plasma_normal = None

            bnorm_pred = S.get_b_field(xyz_plasma=self.Sp.xyz, plasma_normal=plasma_normal)

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
        return RegCoilSolver.from_surfaces(
            S=S,
            Sp=self.Sp,
            bnorm=self.bnorm,
            fit_b_3d=self.fit_b_3d,
            normal_b_field=not self.fit_b_3d,
            use_mu_0_factor=self.use_mu_0_factor,
        )

    def get_results(self, bnorm_pred, phi_mn, S, solver=None, lamb: float = 0.0):
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac
        lamb /= fac**2

        metrics = {}
        results = Results(bnorm_plasma_surface=bnorm_pred, phi_mn_wnet_cur=phi_mn)
        b_err = (bnorm_pred - self.bnorm * fac) ** 2
        metrics["err_max_B"] = np.max(np.abs(b_err))
        if self.fit_b_3d:
            b_err = np.sum(b_err, axis=-1)
        metrics["cost_B"] = self.Sp.integrate(b_err)
        metrics["em_cost"] = metrics["cost_B"]
        if solver is not None:
            if phi_mn is not None:
                metrics["cost_J"] = np.einsum("i,ij,j->", phi_mn, solver.current_basis_dot_prod, phi_mn)
            metrics["em_cost"] = metrics["cost_B"] + lamb * metrics["cost_J"]

        if self.slow_metrics:
            j_3D = S.get_j_3d(phi_mn)
            metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2))
            results.j_3d = j_3D

        return metrics["em_cost"], metrics, results

    def cost_multiple_lambdas(self, S, lambdas):
        solver = self.get_regcoil_solver(S=S)
        metric_results = {}
        results_d = {}
        for lamb in lambdas:
            phi_mn = solver.solve_lambda(lamb=lamb)
            bnorm_pred = solver.biot_et_savart_op.get_b_field(phi_mn)
            metrics, results = self.get_results(bnorm_pred=bnorm_pred, solver=solver, phi_mn=phi_mn, S=S, lamb=lamb)[1:]
            metric_results[float(lamb)] = to_float(metrics)
            results_d[float(lamb)] = results
        return pd.DataFrame(metric_results).T, results_d

    def get_current_weights(self, S):
        solver = self.get_regcoil_solver(S=S)
        return solver.solve_lambda(lamb=self.lamb)

    def get_b_field(self, coil_surf):
        solver = self.get_regcoil_solver(coil_surf)
        phi_mn = solver.solve_lambda(self.lamb)
        bs = self.get_bs_operator(coil_surf, normal_b_field=False)
        return bs.get_b_field(phi_mn)


def to_float(dict_):
    return {k: float(v) for k, v in dict_.items()}


def get_b_field_err(em_cost, coil_surf, err: str = "rmse_n"):
    b_field = em_cost.get_b_field(coil_surf)
    b_field_gt = em_cost.Sp.get_gt_b_field(surface_labels=-1)[:, : em_cost.Sp.integration_par.num_points_v]
    norm_b = np.sqrt(em_cost.Sp.integrate(np.linalg.norm(b_field_gt, axis=-1) ** 2))
    if err == "rmse_n":
        return np.sqrt(em_cost.Sp.integrate(np.linalg.norm(b_field - b_field_gt, axis=-1) ** 2)) / norm_b
    elif err == "max_n":
        return np.max(np.linalg.norm(b_field_gt - b_field, axis=-1)) / np.max(np.linalg.norm(b_field_gt, axis=-1))
