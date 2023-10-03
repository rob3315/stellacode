import numpy as np


def displacement_green_function(xyz_pos, xyz_force, mu: float, nu: float):
    """
    \partial_j U_{i,j}
    where U is the displacement green function
    """
    xyz = xyz_pos - xyz_force
    dist = np.linalg.norm(xyz)
    G = 1 / (16 * (1 - nu) * np.pi * mu * dist) * ((3 - 4 * nu) * np.eye(3) + xyz[:, None] * xyz[None] / dist**2)
    return G


def displacement_green_function(xyz_pos, xyz_force, mu: float, lamb: float):
    """
    \partial_j U_{i,j}
    where G is the stress green function
    """
    xyz = xyz_pos - xyz_force
    dist = np.linalg.norm(xyz)
    nu = get_nu(lamb, mu)
    C = 1 / (16 * (1 - nu) * np.pi * mu)

    xk = xyz[:, None, None]
    xi = xyz[None, :, None]
    xj = xyz[None, None, :]
    return -xk / dist**2 * displacement_green_function(xyz_pos, xyz_force, mu, nu) + C * (
        (np.diag(3)[:, :, None] * xj + np.diag(3)[:, None, :] * xi) / dist**3 - 4 * xi * xj * xk / dist**4
    )


def stress_green_function(xyz_pos, xyz_force, mu: float, lamb: float):
    du = displacement_green_function(xyz_pos, xyz_force, mu, lamb)
    epsilon = du + np.transpose(du, (1, 0, 2))

    sigma = lamb * np.trace(epsilon, axis1=0, axis2=1) * np.eye(3)[:, :, None] + 2 * mu * epsilon
    return sigma


def to_lame(E, nu):
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / 2 * (1 + nu)
    return lamb, mu


def get_nu(lamb, mu):
    return lamb / (2 * (lamb + mu))


def get_E(lamb, mu):
    return mu * (3 * lamb + 2 * mu) / (lamb + mu)
