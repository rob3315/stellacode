import jax
import numpy as onp

from stellacode import np
from stellacode.surface import CoilFactory, Current, IntegrationParams
from stellacode.surface.cylindrical import VerticalCylinder
from stellacode.tools.linear_elasticity import (
    LinearElasticityCoeffs,
    displacement_green_function,
    get_displacement_from_force,
    get_stress_from_force,
    grad_displacement_green_function,
)


def test_green_functions_elasticity():
    """
    Test the green function for computing displacement and stress.

    This function tests the correctness of the green function for linear elasticity.
    It checks the derivative of the green function with respect to the position.
    It also checks the symmetries of the displacement and stress field.
    """

    # Define the material properties
    mu = 1  # shear modulus
    nu = 1e-3  # Poisson's ratio
    coeff = LinearElasticityCoeffs.from_E_nu(mu, nu)  # material coefficients

    # Define the test points
    xyz1 = onp.array([0, 0, 0], dtype=float)  # position
    xyz2 = onp.array([0, 0, 0.1], dtype=float)  # force point

    perturb = 1e-9
    # Compute the derivative of the green function with respect to the position
    grad_displ_0 = (
        displacement_green_function(
            xyz1 + onp.array([perturb, 0, 0]), xyz2, coeff)
        - displacement_green_function(xyz1, xyz2, coeff)
    ) / perturb
    grad_displ_1 = (
        displacement_green_function(
            xyz1 + onp.array([0, perturb, 0]), xyz2, coeff)
        - displacement_green_function(xyz1, xyz2, coeff)
    ) / perturb
    grad_displ_2 = (
        displacement_green_function(
            xyz1 + onp.array([0, 0, perturb]), xyz2, coeff)
        - displacement_green_function(xyz1, xyz2, coeff)
    ) / perturb

    # Compute the gradient of the green function
    grad_displ = grad_displacement_green_function(xyz1, xyz2, coeff)

    # Check the jacobian of the green function
    assert np.max(np.abs(grad_displ_0 - grad_displ[..., 0])) < 1e-6
    assert np.max(np.abs(grad_displ_1 - grad_displ[..., 1])) < 1e-6
    assert np.max(np.abs(grad_displ_2 - grad_displ[..., 2])) < 1e-6

    # Define the surface and the coil
    surf = VerticalCylinder(integration_par=IntegrationParams(
        num_points_u=8, num_points_v=8))()
    coil_factory = CoilFactory(
        current=Current(num_pol=4, num_tor=4, net_currents=np.array([1.0, 0.0])), build_coils=True
    )
    coil = coil_factory(surf)

    # Define the requested points
    xyz_req = np.stack([np.zeros(10), np.zeros(10) + 0.5,
                       np.linspace(0, 1, 10)], axis=-1)

    # Define the requested points with the correct shape
    xy = np.transpose(np.mgrid[0:11, 0:11] / 10 - 0.5, (1, 2, 0))
    xyz_req = np.concatenate([xy, 0.5 * np.ones_like(xy[:, :, :1])], axis=-1)

    # Compute the laplace force
    laplace_force = coil.laplace_force()

    # Compute the stress and displacement fields
    stress, laplace_force = get_stress_from_force(
        coil, xyz_req=xyz_req, force=laplace_force, lin_coeff=coeff, nfp=1)
    displacement, _ = get_displacement_from_force(
        coil, xyz_req=xyz_req, force=laplace_force, lin_coeff=coeff, nfp=1)

    # Check symmetries
    assert np.all(np.abs(displacement[:, :, -1]) < 1e-20)
    assert np.all(
        np.abs(displacement[:, :, 0] - displacement[:, ::-1, 0]) < 1e-20
    )
    assert np.all(
        np.abs(displacement[:, :, 1] - displacement[::-1, :, 1]) < 1e-20
    )
