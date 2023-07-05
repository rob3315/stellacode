from stellacode import np


def to_polar(x, y):
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, phi


def cartesian_to_toroidal(xyz, tore_radius: float, height: float = 0.0):
    xyz_ = xyz - height

    rphiz = cartesian_to_cylindrical(xyz_)
    r_tor, theta = to_polar(rphiz[..., 0] - tore_radius, rphiz[..., 2])
    return np.stack((r_tor, theta, rphiz[..., 1]), axis=-1)


def cartesian_to_cylindrical(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r, phi = to_polar(x, y)
    return np.stack((r, phi, z), axis=-1)


def fourier_transform(coefficients, val):
    angle = val * (np.arange(coefficients.shape[0]) + 1)
    return np.einsum(
        "ab,ab",
        coefficients,
        np.stack((np.cos(angle), np.sin(angle)), axis=1),
    )
