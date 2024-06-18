import typing as tp

import jax
from jax import Array
from pydantic import BaseModel

from stellacode import np


class LinearElasticityCoeffs(BaseModel):
    lamb: float
    mu: float
    nu: float
    E: float

    @classmethod
    def from_lame(cls, lamb: float, mu: float):
        """
        Create `LinearElasticityCoeffs` instance from Lamé parameters.

        Args:
            lamb (float): Lambda parameter.
            mu (float): Mu parameter.

        Returns:
            LinearElasticityCoeffs: Instance of `LinearElasticityCoeffs` with
                calculated Lame parameters.
        """
        # Calculate Poisson's ratio
        nu = lamb / (2 * (lamb + mu))
        # Calculate Young's modulus
        E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
        return cls(lamb=lamb, mu=mu, nu=nu, E=E)

    @classmethod
    def from_E_nu(cls, E: float, nu: float):
        """
        Create `LinearElasticityCoeffs` instance from Young's modulus and Poisson's ratio.

        Args:
            E (float): Young's modulus.
            nu (float): Poisson's ratio.

        Returns:
            LinearElasticityCoeffs: Instance of `LinearElasticityCoeffs` with
                calculated Lamé parameters.
        """
        # Calculate Lambda parameter
        lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
        # Calculate Mu parameter
        mu = E / 2 * (1 + nu)
        return cls(lamb=lamb, mu=mu, nu=nu, E=E)


def displacement_green_function(xyz_pos: Array, xyz_force: Array, lin_coeff: LinearElasticityCoeffs) -> Array:
    """
    Compute the displacement green function.

    Args:
        xyz_pos (Array): Position of the point.
        xyz_force (Array): Force point.
        lin_coeff (LinearElasticityCoeffs): Linear elasticity coefficients.

    Returns:
        Array: Displacement green function.

    The displacement green function `U_{i,k}` is defined as:
    where U is the displacement green function, i is the displacement dimension
    and k is the force dimension.
    """
    # Calculate the vector from the point to the force
    xyz = xyz_pos - xyz_force

    # Calculate the distance between the points
    dist = np.linalg.norm(xyz)

    # Calculate the displacement green function
    G = (
        1
        / (16 * (1 - lin_coeff.nu) * np.pi * lin_coeff.mu * dist)
        * ((3 - 4 * lin_coeff.nu) * np.eye(3) + xyz[:, None] * xyz[None] / dist**2)
    )

    return G


# def grad_displacement_green_function(xyz_pos, xyz_force, mu: float, nu: float):
#     """
#     \partial_k U_{i,j}
#     where U is the displacement green function
#     """
#     xyz = xyz_pos - xyz_force
#     dist = np.linalg.norm(xyz)
#     # nu = get_nu(lamb, mu)
#     C = 1 / (16 * (1 - nu) * np.pi * mu)

#     xk = xyz[:, None, None]
#     xi = xyz[None, :, None]
#     xj = xyz[None, None, :]
#     return -xk / dist**2 * displacement_green_function(xyz_pos, xyz_force, mu, nu)[None, :, :] + C * (
#         (np.eye(3)[:, :,None] * xj + np.eye(3)[:,None, :] * xi) / dist**3 - 4 * xi * xj * xk / dist**4
#     )

# The jacobian adds a dimension along the last dimension
grad_displacement_green_function = jax.jacobian(displacement_green_function)


def strain_green_function(xyz_pos: Array, xyz_force: Array, lin_coeff: LinearElasticityCoeffs) -> Array:
    """
    Calculate the strain green function \Epsilon_ikj.

    Args:
        xyz_pos (Array): Position of the point.
        xyz_force (Array): Force point.
        lin_coeff (LinearElasticityCoeffs): Linear elasticity coefficients.

    Returns:
        Array: Strain green function.

    The strain green function is defined as:
    \Epsilon_ikj where i and j are the strain dimensions and k is the force dimension.
    It is calculated as the average of the displacement green function gradient.
    """
    # Calculate the displacement green function gradient
    du = grad_displacement_green_function(xyz_pos, xyz_force, lin_coeff)

    # Calculate the strain green function by taking the average of the displacement gradient
    epsilon_gf = (du + np.transpose(du, (2, 1, 0))) / 2

    return epsilon_gf


def stress_green_function(xyz_pos: Array, xyz_force: Array, lin_coeff: LinearElasticityCoeffs) -> Array:
    """
    Calculate the stress green function \Sigma_ikj.

    The stress green function is defined as:
    \Sigma_ikj where i and j are the stress dimensions and k is the force dimension.

    Args:
        xyz_pos (Array): Position of the point.
        xyz_force (Array): Force point.
        lin_coeff (LinearElasticityCoeffs): Linear elasticity coefficients.

    Returns:
        Array: Stress green function.
    """
    # Calculate the strain green function
    epsilon_gf = strain_green_function(xyz_pos, xyz_force, lin_coeff)

    # Calculate the trace of the strain green function
    trace = np.trace(epsilon_gf, axis1=0, axis2=2)

    # Calculate the stress green function by combining the trace and the strain green function
    return (
        # Lambda * trace * Identity matrix
        lin_coeff.lamb * trace * np.eye(3)[:, None, :]
        + 2 * lin_coeff.mu * epsilon_gf  # 2 * Mu * Strain green function
    )


def integral_gf_single_point(
    xyz_req: Array, xyz_int: Array, fun: tp.Callable, ds_int: Array, vector_field: Array
) -> Array:
    """
    Calculate the integral of a function `fun` over a set of points.

    Args:
        xyz_req (Array): Requested points.
        xyz_int (Array): Integration points.
        fun (Callable): Function to integrate.
        ds_int (Array): Integration weights.
        vector_field (Array): Vector field at each integration point.

    Returns:
        Array: Integral of the function over the set of points.
    """
    # Calculate the values of the function at each request and integration point
    vals = jax.vmap(fun, in_axes=(None, 0))(xyz_req, xyz_int)

    # Calculate the integral of the function by summing the product of the values, weights, and vector field
    return np.einsum("dikj,dk,d->ij", vals, vector_field, ds_int)


integral_gf = jax.vmap(integral_gf_single_point,
                       in_axes=(0, None, None, None, None))


def integral_gf1_single_point(
    xyz_req: Array,  # Requested points
    xyz_int: Array,  # Integration points
    fun: tp.Callable,  # Function to integrate
    ds_int: Array,  # Integration weights
    vector_field: Array,  # Vector field at each integration point
) -> Array:
    """
    Calculate the integral of a function `fun` over a set of points.

    Args:
        xyz_req (Array): Requested points.
        xyz_int (Array): Integration points.
        fun (Callable): Function to integrate.
        ds_int (Array): Integration weights.
        vector_field (Array): Vector field at each integration point.

    Returns:
        Array: Integral of the function over the set of points.
    """
    # Calculate the values of the function at each request and integration point
    vals = jax.vmap(fun, in_axes=(None, 0))(xyz_req, xyz_int)

    # Calculate the integral of the function by summing the product of the values, weights, and vector field
    return np.einsum("dik,dk,d->i", vals, vector_field, ds_int)


integral_gf1 = jax.vmap(integral_gf1_single_point,
                        in_axes=(0, None, None, None, None))


def get_stress_from_force(
    surf_coil, xyz_req: Array, force: Array, lin_coeff: LinearElasticityCoeffs, nfp: int
) -> tp.Tuple[Array, Array]:
    """
    Calculate the stress tensor at requested points due to a given force.

    Args:
        surf_coil (Surface): Coil surface.
        xyz_req (Array): Requested points. Shape (L, M, 3).
        force (Array): Force acting on the coil surface. Shape (L, M, 3).
        lin_coeff (LinearElasticityCoeffs): Linear elasticity coefficients.
        nfp (int): Number of field periods.

    Returns:
        Tuple[Array, Array]: Stress tensor at requested points, shape (L, M, 3, 3), and force.
    """
    # Get the shapes of the requested points
    lu, lv, _ = xyz_req.shape

    # Calculate the stress tensor at each requested point due to the given force
    stress = integral_gf(
        np.reshape(xyz_req, (-1, 3)),  # Reshape requested points to (D, 3)
        # Reshape integration points to (D, 3)
        np.reshape(surf_coil.xyz, (-1, 3)),
        lambda x, y: stress_green_function(
            x, y, lin_coeff=lin_coeff),  # Stress green function
        np.reshape(surf_coil.ds, -1) * surf_coil.dudv,  # Integration weights
        np.reshape(force, (-1, 3)),  # Reshape force to (D, 3)
    )

    # Reshape the stress tensor to (L, M, 3, 3) and return it along with the force
    return np.reshape(stress, (lu, lv, 3, 3)), force


def get_displacement_from_force(
    surf_coil,
    xyz_req: Array,  # Shape (L, M, 3). Requested points.
    force: Array,  # Shape (L, M, 3). Force acting on the coil surface.
    lin_coeff: LinearElasticityCoeffs,  # Linear elasticity coefficients.
    nfp: int,  # Number of field periods.
) -> tp.Tuple[Array, Array]:
    """
    Calculate the displacement at requested points due to a given force.

    Args:
        surf_coil (Surface): Coil surface.
        xyz_req (Array): Requested points. Shape (L, M, 3).
        force (Array): Force acting on the coil surface. Shape (L, M, 3).
        lin_coeff (LinearElasticityCoeffs): Linear elasticity coefficients.
        nfp (int): Number of field periods.

    Returns:
        Tuple[Array, Array]: Displacement at requested points, shape (L, M, 3), and force.
    """
    # Get the shapes of the requested points
    lu, lv, _ = xyz_req.shape

    # Reshape the requested points and force to (D, 3)
    xyz_req_reshaped = np.reshape(xyz_req, (-1, 3))
    force_reshaped = np.reshape(force, (-1, 3))

    # Reshape the integration points to (D, 3)
    integration_points = np.reshape(surf_coil.xyz, (-1, 3))

    # Calculate the displacement at each requested point due to the given force
    displacement = integral_gf1(
        xyz_req_reshaped,  # Reshape requested points to (D, 3)
        integration_points,  # Reshape integration points to (D, 3)
        lambda x, y: displacement_green_function(
            x, y, lin_coeff=lin_coeff),  # Displacement green function
        np.reshape(surf_coil.ds, -1) * surf_coil.dudv,  # Integration weights
        force_reshaped,  # Reshape force to (D, 3)
    )

    # Reshape the displacement to (L, M, 3) and return it along with the force
    return np.reshape(displacement, (lu, lv, 3)), force
