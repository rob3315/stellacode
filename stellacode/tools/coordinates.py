import numpy as np


def cylindrical_to_cartesian(vector_field: np.ndarray, phi: np.ndarray):
    """Map an array from cylindrical to cartesian coordinates"""
    vr, vphi, vz = vector_field[..., 0], vector_field[..., 1], vector_field[..., 2]
    return np.stack(
        (
            vr * np.cos(phi) - vphi * np.sin(phi),
            vr * np.sin(phi) + vphi * np.cos(phi),
            vz,
        ),
        axis=-1,
    )


def vmec_to_cylindrical(vector_field: np.ndarray, rphiz: np.ndarray, grad_rphiz: np.ndarray):
    """Map an array from vmec to cylindrical coordinates"""
    r_grad = grad_rphiz[..., 0, :]
    z_grad = grad_rphiz[..., 2, :]
    radius = rphiz[..., 0]

    return np.stack(
        (
            np.einsum("tzmc,tzmc->tzm", vector_field[..., :2], r_grad),
            radius * vector_field[..., 1],
            np.einsum("tzmc,tzmc->tzm", vector_field[..., :2], z_grad),
        ),
        axis=-1,
    )
