from scipy.integrate import quad

from stellacode import np


def to_polar(x, y):
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, phi


def from_polar(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


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


def fourier_coefficients(li, lf, n, f):
    # from: https://www.bragitoff.com/2021/05/fourier-series-coefficients-and-visualization-python-program/
    l = (lf - li) / 2
    # Constant term
    a0 = 1 / l * quad(lambda x: f(x), li, lf)[0]
    # Cosine coefficents
    A = np.zeros((n))
    # Sine coefficents
    B = np.zeros((n))

    coefs = []
    for i in range(1, n + 1):
        omega = i * np.pi / l
        A = quad(lambda x: f(x) * np.cos(omega * x), li, lf)[0] / l
        B = quad(lambda x: f(x) * np.sin(omega * x), li, lf)[0] / l
        coefs.append(np.array([A, B]))

    return a0 / 2.0, np.stack(coefs, axis=0)


def fit_to_surface(fitted_surface, surface):
    # Tries to find approximately the smallest fitted_surface enclosing surface
    # assuming surface has get_major_radius and get_minor_radius methods

    major_radius = surface.get_major_radius()
    minor_radius = surface.get_minor_radius()
    new_surf = fitted_surface.copy()
    new_surf.update_params(
        radius=minor_radius + major_radius / 3,
        distance=major_radius,
    )

    min_dist = new_surf.get_min_distance(surface.P)

    new_surf.update_params(
        radius=minor_radius + major_radius / 3 - min_dist,
        distance=major_radius,
    )

    return new_surf
