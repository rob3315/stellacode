import configparser

import jax

jax.config.update("jax_enable_x64", True)
import pytest
from scipy.io import netcdf_file

from stellacode import np
from stellacode.costs.em_cost import EMCost, get_b_field_err
from stellacode.definitions import w7x_plasma, ncsx_plasma
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface import CylindricalSurface, FourierSurface
from stellacode.surface.imports import (
    get_current_potential,
    get_cws,
    get_net_current,
    get_plasma_surface,
    get_cws_from_plasma_config,
)
from stellacode.surface.rotated_surface import RotatedCoil, RotatedSurface, RotateNTimes, ConcatSurfaces
from stellacode.surface.utils import fit_to_surface
from stellacode.tools.vmec import VMECIO
from stellacode.surface.coil_surface import CoilSurface
from stellacode.costs import (
    AggregateCost,
    CurrentCtrCost,
    DistanceCost,
    EMCost,
    NegTorCurvatureCost,
)
from stellacode.costs.em_cost import MSEBField
from stellacode.costs.utils import Constraint
from stellacode.definitions import w7x_plasma
from stellacode.optimizer import Optimizer
from stellacode.surface import IntegrationParams


def test_non_axisymmetric_cylinders():
    n_harmonics = 4
    factor = 6
    num_points = n_harmonics * factor
    method = "quadratic"

    em_cost = MSEBField.from_plasma_config(
        plasma_config=w7x_plasma,
        integration_par=IntegrationParams(num_points_u=num_points, num_points_v=num_points),
    )

    distance = DistanceCost(
        Sp=em_cost.Sp,
        constraint=Constraint(limit=0.2, distance=0.03, minimum=True, method=method),
    )
    current_ctr = CurrentCtrCost(constraint=Constraint(limit=9, distance=0.3, minimum=False, method=method))
    neg_curv = NegTorCurvatureCost(constraint=Constraint(limit=-0.05, distance=0.1, minimum=True, method=method))
    agg_cost = AggregateCost(costs=[em_cost, distance, neg_curv, current_ctr])

    num_tor_symmetry = VMECIO.from_grid(w7x_plasma.path_plasma).nfp
    num_cyl = 3
    num_sym_by_cyl = num_tor_symmetry * num_cyl
    angle = 2 * np.pi / num_sym_by_cyl

    surfaces = []
    for n in range(num_cyl):
        current = CurrentZeroTorBC(
            num_pol=n_harmonics,
            num_tor=n_harmonics,
            sin_basis=True,
            cos_basis=True,
            net_currents=get_net_current(w7x_plasma.path_plasma),
        )
        fourier_coeffs = np.zeros((5, 2))
        surface = CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            integration_par=IntegrationParams(num_points_u=num_points, num_points_v=num_points),
            num_tor_symmetry=num_sym_by_cyl,
        )
        coil_surf = CoilSurface(surface=surface, current=current)
        coil_surf = RotatedSurface(
            surface=coil_surf,
            rotate_n=RotateNTimes(angle=angle, max_num=n + 1, min_num=n),
        )
        surfaces.append(coil_surf)
    coil_surf = ConcatSurfaces(surfaces=surfaces)

    opt = Optimizer.from_cost(
        agg_cost,
        coil_surf,
        method="L-BFGS-B",
        kwargs=dict(options={"disp": True, "maxls": 10, "maxiter": 5}),
    )

    cost, metrics, results, optimized_params = opt.optimize()

    metrics["deltaB_B"] = np.sqrt(metrics["cost_B"] / 1056)

    assert metrics["deltaB_B"] < 0.15
