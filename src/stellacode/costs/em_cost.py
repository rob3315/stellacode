from os.path import join
import typing as tp

import pandas as pd
from jax import Array
from jax.typing import ArrayLike
from pydantic import BaseModel

from stellacode import mu_0_fac, np, PROJECT_PATH
from stellacode.costs.abstract_cost import AbstractCost, Results
from stellacode.definitions import PlasmaConfig
from stellacode.surface import FourierSurfaceFactory, IntegrationParams, Surface
from stellacode.surface.imports import get_plasma_surface
from stellacode.tools.bnorm import get_bnorm
from stellacode.tools.vmec import VMECIO
from stellacode.surface.coil_surface import CoilOperator, CoilSurface

from .utils import merge_dataclasses


class BiotSavartOperator(BaseModel):
    """
    Represent a biot et savart operator with magnetic constant

    Args:
        * bs: biot est savart operator tensor
        * use_mu_0_factor: whether to scale the operator tensor
    """

    model_config = dict(arbitrary_types_allowed=True)

    bs: Array
    use_mu_0_factor: bool = False

    def get_b_field(self, phi_mn):
        """
        Compute the magnetic field given the fourier coefficients

        Args:
            * phi_mn: fourier coefficients

        Returns:
            * b_field: magnetic field tensor
        """

        # Compute dimensions of the magnetic field tensor
        dims = "pqabcd"[: len(self.bs.shape) - 1]

        # Compute the magnetic field tensor
        b_field = np.einsum(f"h{dims},h->{dims}", self.bs, phi_mn)

        # If mu_0 factor is not used, scale the magnetic field tensor
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
        * biot_et_savart_op: biot et savard operator used to build the problem
        * use_mu_0_factor: whether the magnetic constant is used in chi_B
    """

    model_config = dict(arbitrary_types_allowed=True)

    current_basis_dot_prod: ArrayLike
    matrix: ArrayLike
    matrix_reg: ArrayLike
    rhs: ArrayLike
    rhs_reg: ArrayLike
    net_currents: ArrayLike
    biot_et_savart_op: BiotSavartOperator
    use_mu_0_factor: bool = True

    def solve_lambda(self, lamb: float = 0.0):
        """
        Solve regcoil for a given lambda value :
        min (chi_B^2+lambda*chi_J^2)

        Args:
            lamb: lambda value

        Returns:
            phi_mn: weights of the current basis functions
        """
        # If mu_0 factor is not used, scale chi_J accordingly
        if not self.use_mu_0_factor:
            lamb /= mu_0_fac**2

        # Compute the right-hand side of the equation
        if self.rhs_reg is not None:
            rhs = self.rhs - lamb * self.rhs_reg
        else:
            rhs = self.rhs

        # Solve the equation
        phi_mn = np.linalg.solve(self.matrix + lamb * self.matrix_reg, rhs)

        # Concatenate the net currents with phi_mn if net currents are given
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
        """
        Compute RegCoilSolver object from plasma and coil surfaces.
        Source : "Optimal shape of stellarators for magnetic confinement fusion"

        Args:
            S: plasma surface
            Sp: coil surface
            bnorm: normal magnetic field on the plasma surface
            normal_b_field: whether to project the magnetic field on the normal component
            use_mu_0_factor: whether to use the magnetic constant
            fit_b_3d: whether to fit the magnetic field in 3D

        Returns:
            RegCoilSolver object
        """
        # Compute the magnetic field operator
        if normal_b_field:
            kwargs = dict(plasma_normal=Sp.normal_unit)
        else:
            kwargs = {}
        BS = S.get_b_field_op(xyz_plasma=Sp.xyz, **kwargs,
                              use_mu_0_factor=use_mu_0_factor)

        # Create a BiotSavartOperator object
        bs = BiotSavartOperator(bs=BS, use_mu_0_factor=use_mu_0_factor)
        BS = bs.bs

        # Compute the current basis dot product
        current_basis_dot_prod = S.get_current_basis_dot_prod()

        # Compute the dimensions for the 3D magnetic field
        if fit_b_3d:
            add_dim = "a"
        else:
            add_dim = ""

        # Compute the right-hand side of the equation
        if S.net_currents is not None:
            BS_R = BS[2:]
            bnorm_ = bnorm - \
                np.einsum(f"tpq{add_dim},t->pq{add_dim}",
                          BS[:2], S.net_currents)
        else:
            BS_R = BS
            bnorm_ = bnorm
        BS_dagger = np.einsum(
            f"uij{add_dim},ij->uij{add_dim}", BS_R, Sp.ds / Sp.npts)

        rhs = np.einsum(f"hpq{add_dim},pq{add_dim}->h", BS_dagger, bnorm_)

        if S.net_currents is not None:
            rhs_reg = current_basis_dot_prod[2:, :2] @ S.net_currents
        else:
            rhs_reg = None

        # Compute the matrix of the equation
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
        * use_mu_0_factor: Whether to use the magnetic constant
        * slow_metrics: compute metrics that take time.
        * train_currents: Whether to use the updated parameters to compute the currents or RegCoil
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
    ) -> "MSEBField":
        """
        Create an instance of MSEBField from a plasma configuration.

        Args:
            plasma_config (PlasmaConfig): Plasma configuration.
            integration_par (IntegrationParams): Integration parameters.
            Sp (Surface, optional): Plasma surface. If None, it is created from the plasma configuration.
            use_mu_0_factor (bool, optional): Whether to use the magnetic constant. Defaults to True.
            fit_b_3d (bool, optional): Whether to fit the 3D field instead of the normal magnetic field. Defaults to False.
            surface_label (int, optional): Label of the surface. Defaults to -1.

        Returns:
            MSEBField: An instance of MSEBField.
        """
        if Sp is None:
            # Create a Fourier surface from the plasma configuration
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
            # Scale the normal magnetic field accordingly
            if use_mu_0_factor:
                factor = 1
            else:
                factor = 1/mu_0_fac
            bnorm_ = - \
                vmec.scale_bnorm(
                    get_bnorm(plasma_config.path_bnorm, Sp), factor=factor)
        else:
            bnorm_ = 0.0

        if fit_b_3d:
            # Fit the 3D magnetic field instead of the normal magnetic field
            bnorm_ = vmec.b_cartesian[-1]

        return cls(
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
            fit_b_3d=fit_b_3d,
        )

    def cost(self, S, results: Results = Results()):
        """
        Compute the cost function for the MSEBField.

        Args:
            S: The Surface object.
            results (Results, optional): The Results object. Defaults to Results().

        Returns:
            Tuple[float, Dict[str, float], Results, Surface]: The cost, metrics, results, and S.
        """
        # Compute the normal plasma normal, if not fitting the 3D magnetic field
        if not self.fit_b_3d:
            plasma_normal = self.Sp.normal_unit
        else:
            plasma_normal = None

        # Compute the predicted normal magnetic field at the plasma surface
        bnorm_pred = S.get_b_field(
            xyz_plasma=self.Sp.xyz, plasma_normal=plasma_normal, use_mu_0_factor=True)

        # Compute the results and metrics
        cost, metrics, results_ = self.get_results(bnorm_pred=bnorm_pred, S=S)

        # Return the cost, metrics, merged results, and S
        return cost, metrics, merge_dataclasses(results, results_), S

    def get_results(self, bnorm_pred, S):
        """
        Compute the results and metrics for the EMCost.

        Args:
            * bnorm_pred : The predicted normal magnetic field at the plasma surface
            * S : The Surface object

        Returns:
            Tuple[float, Dict[str, float], Results]: The cost, metrics, and results
        """
        # Compute the factor for the normal magnetic field
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac

        # Initialize the metrics dictionary
        metrics = {}

        # Initialize the results object
        results = Results(bnorm_plasma_surface=bnorm_pred)

        # Compute the error in the normal magnetic field
        b_err = (bnorm_pred - self.bnorm*fac) ** 2
        metrics["err_max_B"] = np.max(np.abs(b_err))

        # If fitting the 3D magnetic field, sum the error in all directions
        if self.fit_b_3d:
            b_err = np.sum(b_err, axis=-1)

        # Compute the cost in the normal magnetic field
        metrics["cost_B"] = self.Sp.integrate(b_err)

        # Set the overall cost to the cost in the normal magnetic field
        metrics["cost"] = metrics["cost_B"]

        # If computing slow metrics, compute the maximum current and the 3D current
        if self.slow_metrics:
            j_3D = S.j_3d
            metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2))
            results.j_3d = j_3D

        # Return the cost, metrics, and results
        return metrics["cost"], metrics, results


class EMCost(AbstractCost):
    """
    Cost of the optimal current inverse problem
    chi^2 = chi_B^2 + lambda chi_J^2

    Args:
        * lamb: Regularization parameter lambda
        * Sp: Plasma surface
        * bnorm: Normal magnetic field on the surface pointing inside the plasma
        * use_mu_0_factor: Wether to take the mu_0/4pi factor into account in the Biot et Savart operator
        * slow_metrics: Compute metrics that take time
        * train_currents: Take the chi_J into account
        * fit_b_3d: Regression of the 3D field instead of the normal one
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
        """
        Create an instance of EMCost from a configuration.

        Args:
            config (dict): The configuration.
            Sp (Surface, optional): The plasma surface. If None, it is created from the configuration.
            use_mu_0_factor (bool, optional): Whether to take the mu_0/4pi factor into account in the Biot et Savart operator.

        Returns:
            EMCost: An instance of EMCost.
        """
        # Extract the current polarization from the configuration
        curpol = float(config["other"]["curpol"])

        # If Sp is None, create a plasma surface from the configuration
        if Sp is None:
            Sp = get_plasma_surface(config)()

        # Compute the normal magnetic field on the surface pointing inside the plasma
        bnorm_ = -curpol * get_bnorm(
            join(PROJECT_PATH, str(config["other"]["path_bnorm"])), Sp)

        # If not using the mu_0/4pi factor, scale the normal magnetic field accordingly
        if not use_mu_0_factor:
            bnorm_ /= mu_0_fac

        # Create an instance of EMCost with the extracted parameters
        return cls(
            lamb=float(config["other"]["lamb"]) /
            int(config["geometry"]["Np"]),
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
        """
        Create an instance of EMCost from a plasma configuration.

        Args:
            plasma_config (PlasmaConfig): Plasma configuration.
            integration_par (IntegrationParams): Integration parameters.
            Sp (Surface, optional): Plasma surface. If None, it is created from the plasma configuration.
            use_mu_0_factor (bool, optional): Whether to use the magnetic constant. Defaults to True.
            lamb (float, optional): Regularization parameter. Defaults to 0.1.
            train_currents (bool, optional): Whether to train the currents. Defaults to False.
            fit_b_3d (bool, optional): Whether to fit the 3D field instead of the normal magnetic field. Defaults to False.
            surface_label (int, optional): Label of the surface. Defaults to -1.

        Returns:
            EMCost: An instance of EMCost.
        """
        # Create a Fourier surface from the plasma configuration if Sp is None
        if Sp is None:
            Sp = FourierSurfaceFactory.from_file(
                plasma_config.path_plasma, integration_par=integration_par)()

        # Load the VMECIO grid from the plasma configuration
        vmec = VMECIO.from_grid(
            plasma_config.path_plasma,
            ntheta=integration_par.num_points_u,
            nzeta=integration_par.num_points_v,
            surface_label=surface_label,
        )

        # Scale the normal magnetic field if path_bnorm is provided
        if plasma_config.path_bnorm is not None:
            if use_mu_0_factor:
                factor = 1
            else:
                # scale bnorm accordingly
                factor = 1/mu_0_fac
            bnorm_ = - \
                vmec.scale_bnorm(
                    get_bnorm(plasma_config.path_bnorm, Sp), factor=factor)
        else:
            bnorm_ = 0.0

        # Fit the 3D field if fit_b_3d is True
        if fit_b_3d:
            bnorm_ = vmec.b_cartesian[-1]

        # Create an instance of EMCost with the extracted parameters
        return cls(
            lamb=lamb,
            bnorm=bnorm_,
            Sp=Sp,
            use_mu_0_factor=use_mu_0_factor,
            train_currents=train_currents,
            fit_b_3d=fit_b_3d,
        )

    def cost(self, S, results: Results = Results()):
        """
        Compute the cost function for the given coil surface.

        Args:
            S (CoilOperator or CoilSurface): The coil surface to compute the cost for.
            results (Results, optional): The results object to store the computed metrics. Defaults to an empty Results object.

        Returns:
            tuple: A tuple containing the cost, computed metrics, the merged results object, and the original coil surface.
        """

        # If train_currents is False, solve for the current potential using the regcoil solver and compute the predicted
        # normal magnetic field using the Biot-Savart law.
        if not self.train_currents:
            assert isinstance(S, CoilOperator)
            solver = self.get_regcoil_solver(
                S=S)
            phi_mn = solver.solve_lambda(
                lamb=self.lamb)
            bnorm_pred = solver.biot_et_savart_op.get_b_field(phi_mn)
        else:
            # If train_currents is True, get the coil surface directly from the input and compute the predicted
            # normal magnetic field using the get_b_field method.
            if isinstance(S, CoilOperator):
                coil_surface = S.get_coil()
            else:
                coil_surface = S

            phi_mn = None

            # If fit_b_3d is False, compute the predicted normal magnetic field using the get_b_field method and
            # the plasma normal. Otherwise, compute the predicted normal magnetic field directly.
            if not self.fit_b_3d:
                plasma_normal = self.Sp.normal_unit
            else:
                plasma_normal = None

            bnorm_pred = coil_surface.get_b_field(
                xyz_plasma=self.Sp.xyz, plasma_normal=plasma_normal, use_mu_0_factor=True)

            solver = None

        # Compute the cost and metrics using the predicted normal magnetic field and the input coil surface.
        cost, metrics, results_ = self.get_results(
            bnorm_pred=bnorm_pred, solver=solver, phi_mn=phi_mn, S=S, lamb=self.lamb
        )

        # Merge the computed metrics with the input results object and return the cost, merged results object, and
        # the original coil surface.
        return cost, metrics, merge_dataclasses(results, results_), S

    def get_bs_operator(self, S, normal_b_field: bool = True):
        """
        Compute the Biot-Savart operator.

        Args:
            S: The coil surface.
            normal_b_field: Whether to project the magnetic field on the normal component.

        Returns:
            BiotSavartOperator: The Biot-Savart operator.
        """
        # If normal_b_field is True, compute the plasma normal vector and create a dictionary of keyword arguments.
        # Otherwise, create an empty dictionary of keyword arguments.
        if normal_b_field:
            kwargs = dict(plasma_normal=self.Sp.normal_unit)
        else:
            kwargs = {}

        # Compute the magnetic field operator using the coil surface's get_b_field_op method.
        BS = S.get_b_field_op(xyz_plasma=self.Sp.xyz, **
                              kwargs, use_mu_0_factor=self.use_mu_0_factor)

        # Create a BiotSavartOperator object with the computed magnetic field operator and the use_mu_0_factor flag.
        return BiotSavartOperator(bs=BS, use_mu_0_factor=self.use_mu_0_factor)

    def get_regcoil_solver(self, S):
        """
        Compute RegCoilSolver object from plasma and coil surfaces.

        Args:
            S: The plasma surface.

        Returns:
            RegCoilSolver: The RegCoilSolver object.
        """
        # Compute the RegCoilSolver object from the plasma and coil surfaces using the specified parameters.
        return RegCoilSolver.from_surfaces(
            S=S,  # The plasma surface.
            Sp=self.Sp,  # The coil surface.
            # The normal magnetic field on the plasma surface.
            bnorm=self.bnorm,
            fit_b_3d=self.fit_b_3d,  # Whether to fit the magnetic field in 3D.
            # Whether to project the magnetic field on the normal component.
            normal_b_field=not self.fit_b_3d,
            # Whether to use the magnetic constant.
            use_mu_0_factor=self.use_mu_0_factor,
        )

    def get_results(self, bnorm_pred, phi_mn, S, solver=None, lamb: float = 0.0):
        """
        Compute the cost and return the cost, other metrics, and the results.

        Args:
            bnorm_pred (ndarray): The optimized normal magnetic field.
            phi_mn (ndarray): The optimized Fourier coefficients for the current potential.
            S (object): The optimized CWS.
            solver (object, optional): The RegCoilSolver object. Defaults to None.
            lamb (float, optional): The regularization parameter. Defaults to 0.0.

        Returns:
            tuple: A tuple containing the cost, metrics, and results.
        """
        # Compute the factor for the normal magnetic field
        if self.use_mu_0_factor:
            fac = 1
        else:
            fac = mu_0_fac
        lamb /= fac**2

        # Initialize the metrics dictionary
        metrics = {}

        # Initialize the results object
        results = Results(bnorm_plasma_surface=bnorm_pred,
                          phi_mn_wnet_cur=phi_mn)

        # Compute the error in the normal magnetic field
        b_err = (bnorm_pred - self.bnorm*fac) ** 2
        metrics["max_deltaB_normal"] = np.max(b_err)

        if isinstance(S, CoilOperator):
            metrics["deltaB_B_L2"] = get_b_field_err(self, S, err="L2")
            metrics["deltaB_B_max"] = get_b_field_err(
                self, S, err="max")

        if self.fit_b_3d:
            b_err = np.sum(b_err, axis=-1)

        # Compute the cost in the normal magnetic field
        metrics["cost_B"] = self.Sp.integrate(b_err)

        # Set the overall cost to the cost in the normal magnetic field
        metrics["em_cost"] = metrics["cost_B"]

        # If a solver object is provided and phi_mn is not None, compute the cost in the current basis
        if solver is not None and phi_mn is not None:
            metrics["cost_J"] = np.einsum(
                "i,ij,j->", phi_mn, solver.current_basis_dot_prod, phi_mn)
            metrics["em_cost"] += lamb*metrics["cost_J"]

        # If computing slow metrics, compute the maximum current and the 3D current
        if self.slow_metrics:
            if isinstance(S, CoilOperator):
                j_3D = S.get_j_3d(phi_mn)
            else:
                j_3D = S.j_3d
            metrics["max_j"] = np.max(np.linalg.norm(j_3D, axis=2))
            results.j_3d = j_3D

        # Return the cost, metrics, and results
        return metrics["em_cost"], metrics, results

    def cost_multiple_lambdas(self, S, lambdas):
        """
        Solve for different values of lambda and compute the cost and metrics.

        Args:
            S (CoilOperator or CoilSurface): The magnetic surface.
            lambdas (array-like): The values of lambda for which to compute the cost.

        Returns:
            tuple: A tuple containing a pandas DataFrame with the metrics and a dictionary
                   with the results for each value of lambda.
        """
        # solve for different values of lambda
        solver = self.get_regcoil_solver(
            S=S)

        # Dictionary to store the metrics for each value of lambda
        metric_results = {}
        # Dictionary to store the results for each value of lambda
        results_d = {}

        # Compute the cost and metrics for each value of lambda
        for lamb in lambdas:
            # Solve for the value of phi_mn
            phi_mn = solver.solve_lambda(lamb=lamb)
            # Compute the predicted normal magnetic field
            bnorm_pred = solver.biot_et_savart_op.get_b_field(phi_mn)
            # Compute the metrics and results
            metrics, results = self.get_results(
                bnorm_pred=bnorm_pred, solver=solver, phi_mn=phi_mn, S=S, lamb=lamb)[1:]
            # Store the metrics and results
            metric_results[float(lamb)] = to_float(metrics)
            results_d[float(lamb)] = results

        # Return the metrics and results as a pandas DataFrame and a dictionary
        return pd.DataFrame(metric_results).T, results_d

    def get_current_weights(self, S):
        """
        Compute the current weights for a given magnetic surface.

        Args:
            S (CoilOperator or CoilSurface): The magnetic surface.

        Returns:
            array-like: The current weights.
        """
        # Create a RegCoilSolver for the magnetic surface
        solver = self.get_regcoil_solver(
            S=S)

        # Solve for the current weights using the lambda value
        return solver.solve_lambda(lamb=self.lamb)

    def get_b_field(self, coil_surf):  # unused ?
        """
        Compute the magnetic field on a given coil surface.

        Args:
            coil_surf (CoilSurface): The coil surface on which to compute the magnetic field.

        Returns:
            array-like: The magnetic field on the coil surface.
        """
        # Create a RegCoilSolver for the coil surface
        solver = self.get_regcoil_solver(
            coil_surf)

        # Solve for the value of phi_mn
        phi_mn = solver.solve_lambda(self.lamb)

        # Create a BiotSavartOperator for the coil surface
        bs = self.get_bs_operator(
            coil_surf, normal_b_field=False)

        # Compute the magnetic field on the coil surface using the Biot-Savart law
        return bs.get_b_field(phi_mn)


def to_float(dict_):
    """
    Convert values in a dictionary to float.

    Args:
        dict_ (dict): The dictionary to convert.

    Returns:
        dict: The dictionary with values converted to float.

    """
    # Iterate over the key-value pairs in the dictionary
    return {k: float(v) for k, v in dict_.items()}


def get_b_field_err(em_cost, coil_surface, err: str = "L2"):
    """
    Compute the L2 norm of the difference or maximum relative error between the computed magnetic field and the ground truth magnetic field on a flux surface.

    Args:
        em_cost (EMCost): The EMCost object used to compute the magnetic field.
        coil_surface (CoilSurface): The coil surface on which to compute the magnetic field.
        err (str, optional): The type of error to compute. Defaults to "L2".

    Returns:
        float: The L2 norm of the difference or maximum relative error between the computed and ground truth magnetic field.
    """
    # Compute the computed magnetic field on the coil surface
    b_field = em_cost.get_b_field(coil_surface)

    # Compute the ground truth magnetic field on the coil surface
    b_field_gt = em_cost.Sp.get_gt_b_field(
        surface_labels=-1)[:, : em_cost.Sp.integration_par.num_points_v]

    # Compute the module of the difference between the computed and ground truth magnetic field
    delta_b_module = np.linalg.norm(b_field - b_field_gt, axis=-1)

    # Compute the module of the ground truth magnetic field
    b_field_module = np.linalg.norm(b_field_gt, axis=-1)

    # Compute the L2 norm of the difference or maximum relative error
    if err == "L2":
        # Compute the L2 norm of the difference between the computed and ground truth magnetic field
        return np.sqrt(em_cost.Sp.integrate(delta_b_module ** 2/b_field_module ** 2))
    elif err == "max":
        # Compute the maximum relative error between the computed and ground truth magnetic field
        return np.max(delta_b_module / b_field_module)
