import jax

jax.config.update("jax_enable_x64", True)
from coil_optim.dist_dep2 import get_em_cost
from stellacode.definitions import w7x_plasma
from stellacode.costs import (
    AggregateCost,
    DistanceCost,
    NegTorCurvatureCost,
    CurrentCtrCost,
    LaplaceForceCost,
    PoloidalCurrentCost,
)
from stellacode.tools.vmec import VMECIO
from stellacode.optimizer import Optimizer
from stellacode.costs.utils import Constraint
from coil_optim.dist_dep2 import get_normalization
from stellacode.surface import FourierSurfaceFactory
from stellacode.surface.factories import FreeCylinders

import numpy as np
import pytest


@pytest.mark.parametrize("train_currents", [True, False])
def test_free_cylinders_optim(train_currents):
    """
    Run optimization of cylinders where each cylinder has a different cross section and distance to the center.

    """

    config = w7x_plasma

    n_harmonics_u = 4  # Number of current harmonics in the poloidal direction
    n_harmonics_v = 4  # Number of current harmonics in the toroidal direction
    factor = 6  # Multiplicative factor to get the number of grid points
    num_points = n_harmonics_u * factor
    minor_radius = 0.1
    major_radius = 6  # 0.4
    num_cyl_per_fp = 2  # number of cylinders per plasma field period
    method = "quadratic"  # method used for constraint costs
    current_limit = None
    laplace_force_limit = None
    dist_val = 0.15  # 0.01 # minimum plasma coil distance
    maxiter = 5  # maximum number of iterations
    maxls = 30  # maximum number of line search steps
    scale_phi_mn = 1e8
    add_poloidal_ctr = False  # Add a constraint on the poloidal current to avoid whirlpools of currents
    target_mag_field = 9.13  # 0.1

    em_cost = get_em_cost(config, num_points=num_points, train_currents=train_currents, fit_b_3d=False)
    em_cost.lamb = 1e-25
    plasma_factory = FourierSurfaceFactory.from_file(config.path_plasma, integration_par=em_cost.Sp.integration_par)

    # Get the current division factor to obtain the target magnetic field
    vmec = VMECIO.from_grid(
        em_cost.Sp.file_path,
        ntheta=em_cost.Sp.integration_par.num_points_u,
        nzeta=em_cost.Sp.integration_par.num_points_v,
        surface_label=1,
    )
    divide_mag_field = np.mean(np.linalg.norm(vmec.b_cartesian, axis=-1)) / target_mag_field

    minor_radius = vmec.get_var("Rmajor_p") / vmec.get_var("aspect")

    plasma_factory.Rmn *= major_radius / vmec.get_var("Rmajor_p")
    plasma_factory.Zmn *= major_radius / vmec.get_var("Rmajor_p")
    em_cost.Sp = plasma_factory()

    distance = DistanceCost(
        Sp=em_cost.Sp,
        constraint=Constraint(limit=dist_val * minor_radius, distance=0.03, minimum=True, method=method),
    )

    neg_curv = NegTorCurvatureCost(constraint=Constraint(limit=-0.05, distance=0.1, minimum=True, method=method))

    costs = [em_cost, distance, neg_curv]
    if laplace_force_limit is not None:
        costs.append(
            LaplaceForceCost(
                constraint=Constraint(
                    limit=laplace_force_limit - laplace_force_limit * 0.03,
                    distance=laplace_force_limit * 0.03,
                    minimum=False,
                    method=method,
                ),
                num_tor_symmetry=em_cost.Sp.num_tor_symmetry,
            )
        )
    if current_limit is not None:
        current_ctr = CurrentCtrCost(
            constraint=Constraint(
                limit=current_limit - current_limit * 0.03, distance=current_limit * 0.03, minimum=False, method=method
            )
        )
        costs.append(current_ctr)

    if add_poloidal_ctr:
        pol_curr_ctr = PoloidalCurrentCost(
            constraint=Constraint(limit=0.0, distance=1e5 * 0.03, minimum=True, method=method)
        )
        costs.append(pol_curr_ctr)
    agg_cost = AggregateCost(costs=costs)

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

    # Because the magnetic field is supposed to go from 10T to 0.1T
    coil_factory.net_current /= divide_mag_field

    coil_factory.set_base_current_par(scale_phi_mn=scale_phi_mn)

    # Get the bounds on some parameters
    radii = [k for k in coil_factory.get_trainable_params().keys() if "radius" in k]
    radii_bounds = {k: (em_cost.Sp.get_minor_radius() * 0.8, major_radius * 0.8) for k in radii}
    dist_keys = [k for k in coil_factory.get_trainable_params().keys() if "distance" in k]
    dist_bounds = {k: (major_radius - minor_radius, major_radius + minor_radius) for k in dist_keys}

    opt = Optimizer.from_cost(
        agg_cost,
        coil_factory,
        method="L-BFGS-B",  # from : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        bounds={**dist_bounds, **radii_bounds},
        kwargs=dict(
            options={"disp": True, "maxls": maxls, "maxiter": maxiter},
        ),
    )

    cost, metrics, results, optimized_params = opt.optimize()

    assert metrics["cost_B"] < 400
