from scipy.integrate import quad

from stellacode import np


def to_polar(x, y):
    phi = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return r, phi


def from_polar(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def rotate(x, y, angle):
    return x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)


def cartesian_to_toroidal(xyz, tore_radius: float, height: float = 0.0):
    xyz_ = xyz - height

    rphiz = cartesian_to_cylindrical(xyz_)
    r_tor, theta = to_polar(rphiz[..., 0] - tore_radius, rphiz[..., 2])
    return np.stack((r_tor, theta, rphiz[..., 1]), axis=-1)


def cartesian_to_shifted_cylindrical(xyz, angle: float = 0.0, distance: float = 0.0):
    x_shift, y_shift = from_polar(distance, -angle)
    x, y, z = xyz[..., 0] - x_shift, xyz[..., 1] - y_shift, xyz[..., 2]
    x, y = rotate(x, y, angle)
    r, phi = to_polar(x, z)
    return np.stack((r, phi, y), axis=-1)


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

def unwrap_u(r_function, dr_dphi_function, phi, num_points=100):
    all_phi = np.linspace(0, phi, num_points)

    # Calculate radius and derivative at all phi points
    dr_dphis = np.array([dr_dphi_function(p) for p in all_phi])
    rs = np.array([r_function(p) for p in all_phi])
    integrand = np.sqrt(dr_dphis**2 + rs**2)

    return np.trapz(integrand, all_phi)

def fourier_transform_derivative(coefficients, val):
    k = (np.arange(coefficients.shape[0]) + 1)
    angle = val * k

    cos_terms = np.cos(angle)
    sin_terms = np.sin(angle)
    
    d_cos = -k * sin_terms
    d_sin = k * cos_terms
    
    derivative = np.einsum(
        "ab,ab",
        coefficients,
        np.stack((d_cos, d_sin), axis=1),
    )
    
    return derivative

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


def fit_to_surface(fitted_surface, surface, distance: float = 0.0):
    # Tries to find approximately the smallest fitted_surface enclosing surface
    # assuming surface has get_major_radius and get_minor_radius methods

    major_radius = surface.get_major_radius()
    minor_radius = surface.get_minor_radius()
    new_surf = fitted_surface.model_copy()
    new_surf.surface_factories[0].update_params(
        radius=minor_radius + major_radius / 3,
        distance=major_radius,
    )

    min_dist = new_surf().get_min_distance(surface.xyz)

    new_surf.surface_factories[0].update_params(
        radius=minor_radius + major_radius / 3 - min_dist + distance,
        distance=major_radius,
    )

    return new_surf


def get_principles(hess_xyz, jac_xyz, normal_unit):
    dpsi_uu = hess_xyz[..., 0, 0]
    dpsi_uv = hess_xyz[..., 1, 1]
    dpsi_vv = hess_xyz[..., 0, 1]

    # dNdu = np.cross(dpsi_uu, jac_xyz[1], 0, 0, 0) + np.cross(jac_xyz[0], dpsi_uv, 0, 0, 0)
    # dNdv = np.cross(dpsi_uv, jac_xyz[1], 0, 0, 0) + np.cross(jac_xyz[0], dpsi_vv, 0, 0, 0)
    # dS_u = np.sum(dNdu * N, axis=0) / ds
    # dS_v = np.sum(dNdv * N, axis=0) / ds
    # n_u = dNdu / ds - dS_u * N / (ds**2)
    # n_v = dNdv / ds - dS_v * N / (ds**2)

    # curvature computations :
    # First fundamental form of the surface (E,F,G)
    E = np.einsum("ijl,ijl->ij", jac_xyz[..., 0], jac_xyz[..., 0])
    F = np.einsum("ijl,ijl->ij", jac_xyz[..., 0], jac_xyz[..., 1])
    G = np.einsum("ijl,ijl->ij", jac_xyz[..., 1], jac_xyz[..., 1])

    # Second fundamental of the surface (L,M,N)
    L = np.einsum("ijl,ijl->ij", dpsi_uu, normal_unit)  # e
    M = np.einsum("ijl,ijl->ij", dpsi_uv, normal_unit)  # f
    N = np.einsum("ijl,ijl->ij", dpsi_vv, normal_unit)  # g

    # K = det(second fundamental) / det(first fundamental)
    # Gaussian Curvature
    K = (L * N - M**2) / (E * G - F**2)

    # trace of (second fundamental)(first fundamental^-1)
    # Mean Curvature
    H = ((E * N + G * L - 2 * F * M) / ((E * G - F**2))) / 2

    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)
    return Pmax, Pmin


def get_net_current(coil_surf, toroidal: bool = True):
    j_3d = coil_surf.j_3d
    vec_u = coil_surf.jac_xyz[..., 0]
    vec_v = coil_surf.jac_xyz[..., 1]
    if toroidal:
        int_vec = vec_u
        dl = coil_surf.du
        axis_sum = 0
    else:
        int_vec = vec_v
        dl = coil_surf.dv
        axis_sum = 1

    vecn = np.cross(int_vec, coil_surf.normal, -1, -1, -1)
    vecn = vecn / np.linalg.norm(vecn, axis=-1, keepdims=True)
    j_tor = np.einsum("ija,ija->ij", j_3d, vecn)
    return np.sum(j_tor * np.linalg.norm(int_vec, axis=-1), axis=axis_sum) * dl
