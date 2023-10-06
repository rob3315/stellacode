import jax

jax.config.update("jax_enable_x64", True)

from stellacode import np
from stellacode.definitions import w7x_plasma, ncsx_plasma
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface import CylindricalSurface, FourierSurface
from stellacode.surface.imports import (
    get_net_current,
)
from stellacode.surface.coil_surface import CoilFactory
from stellacode.surface.factory_tools import RotatedSurface, RotateNTimes, ConcatSurfaces, Sequential
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
        constraint=Constraint(limit=0.2, distance=0.2, minimum=True, method=method),
    )
    current_ctr = CurrentCtrCost(constraint=Constraint(limit=100, distance=0.3, minimum=False, method=method))
    neg_curv = NegTorCurvatureCost(constraint=Constraint(limit=-0.05, distance=0.1, minimum=True, method=method))
    agg_cost = AggregateCost(costs=[em_cost, distance, neg_curv, current_ctr])

    nfp = VMECIO.from_grid(w7x_plasma.path_plasma).nfp
    num_cyl = 3
    num_sym_by_cyl = nfp * num_cyl
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
        minor_radius = em_cost.Sp.get_minor_radius()
        major_radius = em_cost.Sp.get_major_radius()
        surface = CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            integration_par=IntegrationParams(num_points_u=num_points, num_points_v=num_points, center_vgrid=True),
            nfp=num_sym_by_cyl,
            radius=minor_radius * 1.5,
            distance=major_radius,
        )
        # surface = em_cost.Sp.get_surface_envelope(num_cyl=3, num_coeff=5, convex=True)
        # surface.update_params(radius=surface.radius*1.5)

        coil_surf = Sequential(
            surface_factories=[
                surface,
                CoilFactory(current=current, build_coils=True),
                RotatedSurface(
                    rotate_n=RotateNTimes(angle=angle, max_num=n + 1, min_num=n),
                ),
            ]
        )
        surfaces.append(coil_surf)

    coil_surf = Sequential(
        surface_factories=[
            ConcatSurfaces(surface_factories=surfaces),
            RotatedSurface(rotate_n=RotateNTimes(angle=2 * np.pi / nfp, max_num=nfp)),
        ]
    )
    opt = Optimizer.from_cost(
        agg_cost,
        coil_surf,
        method="L-BFGS-B",
        kwargs=dict(options={"disp": True, "maxls": 30, "maxiter": 5}),
    )

    cost, metrics, results, optimized_params = opt.optimize()
    # j_3d = coil_surf.get_j_3d()
    # coil_surf.surface.plot(vector_field=j_3d)

    metrics["deltaB_B"] = np.sqrt(metrics["cost_B"] / 1056)

    assert metrics["deltaB_B"] < 0.3
