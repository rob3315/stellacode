import numpy as np


def cylindrical_to_cartesian(vector_field: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Map an array from cylindrical to cartesian coordinates.

    Parameters
    ----------
    vector_field : ndarray
        The vector field in cylindrical coordinates.
    phi : ndarray
        The angle in cylindrical coordinates.

    Returns
    -------
    ndarray
        The vector field in cartesian coordinates.

    Notes
    -----
    The mapping is defined as:
    x = vr * cos(phi) - vphi * sin(phi)
    y = vr * sin(phi) + vphi * cos(phi)
    z = vz
    """
    # Extract the components of the vector field
    vr, vphi, vz = vector_field[...,
                                0], vector_field[..., 1], vector_field[..., 2]

    # Apply the mapping to convert the vector field from cylindrical to cartesian coordinates
    return np.stack(
        (
            vr * np.cos(phi) - vphi * np.sin(phi),
            vr * np.sin(phi) + vphi * np.cos(phi),
            vz,
        ),
        axis=-1,
    )


def vmec_to_cylindrical(vector_field: np.ndarray, rphiz: np.ndarray, grad_rphiz: np.ndarray):
    """
    Map an array from vmec to cylindrical coordinates.

    Parameters
    ----------
    vector_field : ndarray
        The vector field in vmec coordinates.
    rphiz : ndarray
        The cylindrical coordinates.
    grad_rphiz : ndarray
        The gradient of the cylindrical coordinates.

    Returns
    -------
    ndarray
        The vector field in cylindrical coordinates.
    """
    # Extract the components of the gradient of the cylindrical coordinates
    r_grad = grad_rphiz[..., 0, :]
    z_grad = grad_rphiz[..., 2, :]

    # Compute the radius and partial_r
    radius = rphiz[..., 0]

    # Apply the mapping to convert the vector field from vmec to cylindrical coordinates
    return np.stack(
        (
            np.einsum("tzmc,tzmc->tzm",
                      vector_field[..., :2], r_grad),
            radius * vector_field[..., 1],
            np.einsum("tzmc,tzmc->tzm",
                      vector_field[..., :2], z_grad),
        ),
        axis=-1,
    )
