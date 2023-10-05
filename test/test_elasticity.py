from stellacode.tools.linear_elasticity import (
    displacement_green_function,
    grad_displacement_green_function,
    LinearElasticityCoeffs,
    get_stress_from_force,
    get_displacement_from_force,
)
from stellacode.surface.cylindrical import VerticalCylinder
from stellacode.surface import IntegrationParams, CoilFactory, Current
import numpy as onp
from stellacode import np
import jax


def test_green_functions_elasticity():
    mu = 1
    nu = 1e-3
    coeff = LinearElasticityCoeffs.from_E_nu(1, 1e-3)
    xyz1 = onp.array([0, 0, 0], dtype=float)
    xyz2 = onp.array([0, 0, 0.1], dtype=float)
    grad_displ_0 = (
        displacement_green_function(xyz1 + onp.array([1e-9, 0, 0]), xyz2, coeff)
        - displacement_green_function(xyz1, xyz2, coeff)
    ) / 1e-9
    grad_displ_1 = (
        displacement_green_function(xyz1 + onp.array([0, 1e-9, 0]), xyz2, coeff)
        - displacement_green_function(xyz1, xyz2, coeff)
    ) / 1e-9

    grad_displ_2 = (
        displacement_green_function(xyz1 + onp.array([0, 0, 1e-9]), xyz2, coeff)
        - displacement_green_function(xyz1, xyz2, coeff)
    ) / 1e-9
    grad_displ2 = grad_displacement_green_function(xyz1, xyz2, coeff)

    # check jacobian of green function
    assert np.max(np.abs(grad_displ_0 - grad_displ2[:, :, 0])) < 1e-6
    assert np.max(np.abs(grad_displ_1 - grad_displ2[:, :, 1])) < 1e-6
    assert np.max(np.abs(grad_displ_2 - grad_displ2[:, :, 2])) < 1e-6

    surf = VerticalCylinder(integration_par=IntegrationParams(num_points_u=8, num_points_v=8))()
    coil_factory = CoilFactory(
        current=Current(num_pol=4, num_tor=4, net_currents=np.array([1.0, 0.0])), build_coils=True
    )
    coil = coil_factory(surf)

    xyz_req = np.stack([np.zeros(10), np.zeros(10) + 0.5, np.linspace(0, 1, 10)], axis=-1)

    xy = np.transpose(np.mgrid[0:11, 0:11] / 10 - 0.5, (1, 2, 0))
    xyz_req = np.concatenate([xy, 0.5 * np.ones_like(xy[:, :, :1])], axis=-1)

    laplace_force = coil.laplace_force(1)
    stress, laplace_force = get_stress_from_force(
        coil, xyz_req=xyz_req, laplace_force=laplace_force, lin_coeff=coeff, nfp=1
    )
    displacement = get_displacement_from_force(
        coil, xyz_req=xyz_req, laplace_force=laplace_force, lin_coeff=coeff, nfp=1
    )

    # Check symmetries
    assert np.all(np.abs(displacement[:, :, -1]) < 1e-20)
    assert np.all(np.abs(displacement[:, :, 0] - displacement[:, ::-1, 0]) < 1e-20)
    assert np.all(np.abs(displacement[:, :, 1] - displacement[::-1, :, 1]) < 1e-20)

    # import matplotlib.pyplot as plt; import seaborn as sns; sns.heatmap(stress[:,:,0,0], cmap='seismic', center=0);plt.show()
