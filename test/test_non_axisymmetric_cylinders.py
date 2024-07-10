from stellacode.tools.vmec import VMECIO
from stellacode.surface.utils import fit_to_surface
from stellacode.surface.imports import get_net_current
from stellacode.surface.factory_tools import (
    ConcatSurfaces,
    RotatedSurface,
    RotateNTimes,
    Sequential,
)
from stellacode.surface.coil_surface import CoilFactory, CoilSurface
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    CylindricalSurface,
    FourierSurfaceFactory,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.optimizer import Optimizer
from stellacode.definitions import ncsx_plasma, w7x_plasma
from stellacode.costs.utils import Constraint
from stellacode.costs.em_cost import MSEBField
from stellacode.costs import (
    AggregateCost,
    CurrentCtrCost,
    DistanceCost,
    EMCost,
    NegTorCurvatureCost,
)
from stellacode import np
import jax

jax.config.update("jax_enable_x64", True)


def test_non_axisymmetric_cylinders():
    """
    Test non-axisymmetric cylinders optimization.

    This test checks the optimization of num_cyl different cylinders
    per field period, with different cylindrical surfaces for each of them.

    """
    # Define optimization parameters
    n_harmonics = 4  # number of Fourier harmonics for the cylindrical surface
    factor = 6  # factor for the number of sample points
    method = "quadratic"  # method for the distance cost
    num_points = n_harmonics * factor

    # Initialize MSE cost
    em_cost = MSEBField.from_plasma_config(
        plasma_config=w7x_plasma,
        integration_par=IntegrationParams(
            num_points_u=num_points, num_points_v=num_points),
    )

    # Define cost functions
    distance = DistanceCost(
        Sp=em_cost.Sp,
        constraint=Constraint(limit=0.2, distance=0.2,
                              minimum=True, method=method),
    )
    current_ctr = CurrentCtrCost(constraint=Constraint(
        limit=100, distance=0.3, minimum=False, method=method))
    neg_curv = NegTorCurvatureCost(constraint=Constraint(
        limit=-0.05, distance=0.1, minimum=True, method=method))
    agg_cost = AggregateCost(costs=[em_cost, distance, neg_curv, current_ctr])

    # Define cylindrical surfaces
    nfp = w7x_plasma.nfp  # number of field periods
    num_cyl = 3  # number of cylinders per period
    num_sym_by_cyl = nfp * num_cyl  # total number of cylinders
    angle = 2 * np.pi / num_sym_by_cyl  # angular step for a cylinder

    surfaces = []
    for n in range(num_cyl):
        # Define current and cylindrical surface for each cylinder
        current = CurrentZeroTorBC(
            num_pol=n_harmonics,
            num_tor=n_harmonics,
            sin_basis=True,
            cos_basis=True,
            net_currents=get_net_current(w7x_plasma.path_plasma),
        )
        fourier_coeffs = np.zeros((5, 2))
        minor_radius = em_cost.Sp.get_minor_radius(vmec=False)
        major_radius = em_cost.Sp.get_major_radius()
        surface = CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            integration_par=IntegrationParams(
                num_points_u=num_points, num_points_v=num_points, center_vgrid=True),
            ncp=num_sym_by_cyl,
            radius=minor_radius * 1.5,
            distance=major_radius,
        )

        # Construct the coil surface with rotation
        coil_surf = Sequential(
            surface_factories=[
                surface,
                CoilFactory(current=current, build_coils=True),
                RotatedSurface(
                    rotate_n=RotateNTimes(
                        angle=angle, max_num=n + 1, min_num=n),
                ),
            ]
        )
        surfaces.append(coil_surf)

    # Concatenate and rotate the cylindrical surfaces
    coil_surf = Sequential(
        surface_factories=[
            ConcatSurfaces(surface_factories=surfaces),
            RotatedSurface(rotate_n=RotateNTimes(
                angle=2 * np.pi / nfp, max_num=nfp)),
        ]
    )

    # Perform optimization
    opt = Optimizer.from_cost(
        agg_cost,
        coil_surf,
        method="L-BFGS-B",
        kwargs=dict(options={"disp": True, "maxls": 30, "maxiter": 5}),
    )

    cost, metrics, results, optimized_params = opt.optimize()

    # Calculate deltaB_B metric
    # cost wrt B normal / weight wrt full B
    B_L2 = em_cost.Sp.integrate(np.linalg.norm(
        em_cost.Sp.get_gt_b_field(), axis=-1)**2)
    metrics["deltaB_B"] = np.sqrt(metrics["cost_B"] / B_L2)

    # Check the optimization results
    assert metrics["deltaB_B"] < 0.5
