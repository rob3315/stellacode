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
        nu = lamb / (2 * (lamb + mu))
        E = mu * (3 * lamb + 2 * mu) / (lamb + mu)
        return cls(lamb=lamb, mu=mu, nu=nu, E=E)

    @classmethod
    def from_E_nu(cls, E: float, nu: float):
        lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / 2 * (1 + nu)
        return cls(lamb=lamb, mu=mu, nu=nu, E=E)


def displacement_green_function(xyz_pos: Array, xyz_force: Array, lin_coeff: LinearElasticityCoeffs) -> Array:
    """
    U_{i,k}
    where U is the displacement green function, i is the displacement dimension and k is the force dimension
    """
    xyz = xyz_pos - xyz_force
    dist = np.linalg.norm(xyz)
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
    strain green function \Epsilon_ikj where i and j are the strain dimensions and k is the force dimension
    """
    du = grad_displacement_green_function(xyz_pos, xyz_force, lin_coeff)
    return (du + np.transpose(du, (2, 1, 0))) / 2


def stress_green_function(xyz_pos: Array, xyz_force: Array, lin_coeff: LinearElasticityCoeffs) -> Array:
    """
    stress green function \Sigma_ikj where i and j are the stress dimensions and k is the force dimension
    """

    epsilon_gf = strain_green_function(xyz_pos, xyz_force, lin_coeff)

    return (
        lin_coeff.lamb * np.trace(epsilon_gf, axis1=0, axis2=2) * np.eye(3)[:, None, :] + 2 * lin_coeff.mu * epsilon_gf
    )


def integral_gf_single_point(
    xyz_req: Array, xyz_int: Array, fun: tp.Callable, ds_int: Array, vector_field: Array
) -> Array:
    vals = jax.vmap(fun, in_axes=(None, 0))(xyz_req, xyz_int)

    return np.einsum("dikj,dk,d->ij", vals, vector_field, ds_int)


integral_gf = jax.vmap(integral_gf_single_point, in_axes=(0, None, None, None, None))


def integral_gf1_single_point(
    xyz_req: Array, xyz_int: Array, fun: tp.Callable, ds_int: Array, vector_field: Array
) -> Array:
    vals = jax.vmap(fun, in_axes=(None, 0))(xyz_req, xyz_int)

    return np.einsum("dik,dk,d->i", vals, vector_field, ds_int)


integral_gf1 = jax.vmap(integral_gf1_single_point, in_axes=(0, None, None, None, None))


def get_stress_from_force(
    surf_coil, xyz_req: Array, force: Array, lin_coeff: LinearElasticityCoeffs, nfp: int
) -> tp.Tuple[Array, Array]:
    lu, lv, _ = xyz_req.shape
    stress = integral_gf(
        np.reshape(xyz_req, (-1, 3)),
        np.reshape(surf_coil.xyz, (-1, 3)),
        lambda x, y: stress_green_function(x, y, lin_coeff=lin_coeff),
        np.reshape(surf_coil.ds, -1) * surf_coil.dudv,
        np.reshape(force, (-1, 3)),
    )
    return np.reshape(stress, (lu, lv, 3, 3)), force


def get_displacement_from_force(
    surf_coil, xyz_req: Array, force: Array, lin_coeff: LinearElasticityCoeffs, nfp: int
) -> tp.Tuple[Array, Array]:
    lu, lv, _ = xyz_req.shape
    displacement = integral_gf1(
        np.reshape(xyz_req, (-1, 3)),
        np.reshape(surf_coil.xyz, (-1, 3)),
        lambda x, y: displacement_green_function(x, y, lin_coeff=lin_coeff),
        np.reshape(surf_coil.ds, -1) * surf_coil.dudv,
        np.reshape(force, (-1, 3)),
    )
    return np.reshape(displacement, (lu, lv, 3))
