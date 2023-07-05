from stellacode import np


def cartesian_to_toroidal(xyz, tore_radius: float, height: float = 0.):
    xyz_ = xyz - height

    x, y, z = xyz_[..., 0], xyz_[..., 1], xyz_[..., 2]
    phi = np.arctan2(y, x)
    R = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z, R)
    r_tor = np.sqrt((R - tore_radius) ** 2 + z**2)

    return np.stack((r_tor, theta, phi), axis=-1)