import pytest
from stellacode import np
from stellacode.surface.factories import FreeCylinders
from stellacode.surface import FourierSurfaceFactory, IntegrationParams
from stellacode.costs.utils import Constraint
from stellacode.optimizer import Optimizer
from stellacode.tools.vmec import VMECIO
from stellacode.costs import (
    AggregateCost,
    DistanceCost,
    NegTorCurvatureCost,
    CurrentCtrCost,
    LaplaceForceCost,
    PoloidalCurrentCost,
    EMCost,
)
from stellacode.definitions import w7x_plasma
import jax

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("train_currents", [True])
# @pytest.mark.parametrize("train_currents", [True, False])
def test_free_cylinders_optim(train_currents):
    """
    Test free cylinders optimization.

    Args:
        train_currents (bool): Whether to train the currents or not.

    """

    # Define the configuration
    config = w7x_plasma

    # Define the parameters
    n_harmonics_u = 4  # Number of current harmonics in the poloidal direction
    n_harmonics_v = 4  # Number of current harmonics in the toroidal direction
    factor = 6  # Multiplicative factor to get the number of grid points
    num_points = n_harmonics_u * factor
    minor_radius = 0.1
    major_radius = 6
    num_cyl_per_fp = 2  # number of cylinders per plasma field period
    method = "quadratic"  # method used for constraint costs
    current_limit = None
    laplace_force_limit = None
    dist_val = 0.15  # minimum plasma coil distance
    maxiter = 5  # maximum number of iterations
    maxls = 30  # maximum number of line search steps
    scale_phi_mn = 1e8
    # Add a constraint on the poloidal current to avoid whirlpools of currents
    add_poloidal_ctr = False
    target_mag_field = 9.13

    # Get the EM cost
    em_cost = EMCost.from_plasma_config(
        plasma_config=config,
        integration_par=IntegrationParams(
            num_points_u=num_points, num_points_v=num_points),
        use_mu_0_factor=False,
        train_currents=train_currents,
        fit_b_3d=False,
    )
    em_cost.lamb = 1e-25
    plasma_factory = FourierSurfaceFactory.from_file(
        config.path_plasma, integration_par=em_cost.Sp.integration_par)

    # Get the current division factor to obtain the target magnetic field
    vmec = VMECIO.from_grid(
        em_cost.Sp.file_path,
        ntheta=em_cost.Sp.integration_par.num_points_u,
        nzeta=em_cost.Sp.integration_par.num_points_v,
        surface_label=1,
    )
    divide_mag_field = np.mean(np.linalg.norm(
        vmec.b_cartesian, axis=-1)) / target_mag_field

    # Adjust the minor and major radii
    minor_radius = vmec.get_var("Rmajor_p") / vmec.get_var("aspect")
    plasma_factory.Rmn *= major_radius / vmec.get_var("Rmajor_p")
    plasma_factory.Zmn *= major_radius / vmec.get_var("Rmajor_p")
    em_cost.Sp = plasma_factory()

    # Define the distance cost
    distance = DistanceCost(
        Sp=em_cost.Sp,
        constraint=Constraint(limit=dist_val * minor_radius,
                              distance=0.03, minimum=True, method=method),
    )

    # Define the negative toroidal curvature cost
    neg_curv = NegTorCurvatureCost(constraint=Constraint(
        limit=-0.05, distance=0.1, minimum=True, method=method))

    # Define the costs
    costs = [em_cost, distance, neg_curv]

    # Add the laplace force cost if specified
    if laplace_force_limit is not None:
        laplace_force = LaplaceForceCost(
            constraint=Constraint(
                limit=laplace_force_limit - laplace_force_limit * 0.03,
                distance=laplace_force_limit * 0.03,
                minimum=False,
                method=method,
            ),
            num_tor_symmetry=em_cost.Sp.num_tor_symmetry,
        )
        costs.append(laplace_force)

    # Add the current control cost if specified
    if current_limit is not None:
        current_ctr = CurrentCtrCost(
            constraint=Constraint(
                limit=current_limit - current_limit * 0.03,
                distance=current_limit * 0.03,
                minimum=False,
                method=method,
            )
        )
        costs.append(current_ctr)

    # Add the poloidal current center cost if specified
    if add_poloidal_ctr:
        pol_curr_ctr = PoloidalCurrentCost(
            constraint=Constraint(
                limit=0.0, distance=1e5 * 0.03, minimum=True, method=method)
        )
        costs.append(pol_curr_ctr)

    # Define the aggregate cost
    agg_cost = AggregateCost(costs=costs)

    # Generate the coil factory
    coil_factory = FreeCylinders.from_plasma(
        surf_plasma=em_cost.Sp,
        distance=dist_val * minor_radius,
        num_cyl=num_cyl_per_fp,
        n_harmonics_u=n_harmonics_u,
        n_harmonics_v=n_harmonics_v,
        factor=factor,
        constrain_tor_current=True,
    )

    # This way we can prevent training some of the parameters
    for surf in coil_factory._get_base_surfaces():
        surf.trainable_params = [
            "fourier_coeffs",
            # "axis_angle", # Axis angle should not change if we want to avoid coils penetrating through each others
            "radius",
            "distance",
        ]

    # Because the magnetic field is supposed to go from 0.1 to 10 T
    coil_factory.net_current /= divide_mag_field

    coil_factory.set_base_current_par(scale_phi_mn=scale_phi_mn)

    # Get the bounds on some parameters
    radii = [k for k in coil_factory.get_trainable_params().keys()
             if "radius" in k]
    radii_bounds = {k: (em_cost.Sp.get_minor_radius() * 0.8,
                        major_radius * 0.8) for k in radii}
    dist_keys = [k for k in coil_factory.get_trainable_params().keys()
                 if "distance" in k]
    dist_bounds = {k: (major_radius - minor_radius,
                       major_radius + minor_radius) for k in dist_keys}

    # Define the optimizer
    opt = Optimizer.from_cost(
        agg_cost,
        coil_factory,
        # from : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        method="L-BFGS-B",
        bounds={**dist_bounds, **radii_bounds},
        kwargs=dict(
            options={"disp": True, "maxls": maxls, "maxiter": maxiter},
        ),
    )

    # Optimize the cost
    cost, metrics, results, optimized_params = opt.optimize()

    # Check if the cost is within a reasonable range
    assert metrics["cost_B"] < 400
