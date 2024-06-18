from scipy.integrate import quad

from stellacode import np
from scipy.spatial.distance import cdist


def to_polar(x, y):
    """
    Convert 2D Cartesian coordinates (x, y) to polar coordinates (r, phi).

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        tuple: A tuple containing the radius (r) and the angle (phi) in radians.
    """
    # Calculate the angle (phi) using the arctan2 function
    phi = np.arctan2(y, x)

    # Calculate the radius (r) using the Pythagorean theorem
    r = np.sqrt(x**2 + y**2)

    return r, phi


def from_polar(r, theta):
    """
    Convert polar coordinates (r, theta) to 2D Cartesian coordinates (x, y).

    Args:
        r (float): The radius.
        theta (float): The angle in radians.

    Returns:
        tuple: A tuple containing the x-coordinate (x) and the y-coordinate (y).
    """
    # Calculate the x-coordinate using the radius and cosine of the angle
    x = r * np.cos(theta)

    # Calculate the y-coordinate using the radius and sine of the angle
    y = r * np.sin(theta)

    return x, y


def rotate(x, y, angle):
    """
    Rotate a point (x, y) by a given angle.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        angle (float): The angle of rotation in radians.

    Returns:
        tuple: A tuple containing the rotated x-coordinate and y-coordinate.
    """
    # Calculate the rotated x-coordinate using the cosine and sine of the angle
    rot_x = x * np.cos(angle) - y * np.sin(angle)

    # Calculate the rotated y-coordinate using the cosine and sine of the angle
    rot_y = x * np.sin(angle) + y * np.cos(angle)

    return rot_x, rot_y


def cartesian_to_cylindrical(xyz):
    """
    Convert 3D Cartesian coordinates to cylindrical coordinates.

    Args:
        xyz (ArrayLike): The 3D Cartesian coordinates.

    Returns:
        ArrayLike: The cylindrical coordinates.
    """
    # Extract the x, y, and z coordinates
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    # Calculate the radial distance and azimuthal angle
    r, phi = to_polar(x, y)

    # Stack the cylindrical coordinates along the last axis
    return np.stack((r, phi, z), axis=-1)


def cartesian_to_toroidal(xyz, tore_radius: float = 0.0, tore_height: float = 0.0):
    """
    Convert 3D Cartesian coordinates to toroidal coordinates.

    Args:
        xyz (ArrayLike): The 3D Cartesian coordinates.
        tore_radius (float, optional): The major radius of the torus. Defaults to 0.0.
        tore_height (float, optional): The height of the torus. Defaults to 0.0.

    Returns:
        ArrayLike: The toroidal coordinates (r_tor, theta, phi).

    Notes:
        The toroidal coordinates (r_tor, theta, phi) are defined as follows:
            - r_tor: The radial distance from the center of the torus cross section to the point.
            - theta: The angular coordinate measured from the center of the torus.
            - phi: The angle measured from the z-axis.
    """
    # Convert Cartesian coordinates to cylindrical coordinates
    rphiz = cartesian_to_cylindrical(xyz)

    # Calculate the toroidal radius and angle
    r_tor, theta = to_polar(
        rphiz[..., 0] - tore_radius, rphiz[..., 2] - tore_height)

    # Stack the toroidal coordinates (r_tor, theta, phi)
    # The resulting array has shape (..., 3)
    return np.stack((r_tor, theta, rphiz[..., 1]), axis=-1)


def cartesian_to_shifted_cylindrical(xyz, angle: float = 0.0, distance: float = 0.0):
    """
    Convert Cartesian coordinates to shifted cylindrical coordinates.

    Args:
        xyz (ArrayLike): The Cartesian coordinates.
        angle (float, optional): The rotation angle in radians. Defaults to 0.0.
        distance (float, optional): The distance along the z-axis. Defaults to 0.0.

    Returns:
        ArrayLike: The shifted cylindrical coordinates.
    """
    # Calculate the shift in the x and y directions
    x_shift, y_shift = from_polar(distance, angle)

    # Remove the shift from the Cartesian coordinates
    x, y, z = xyz[..., 0] - x_shift, xyz[..., 1] - y_shift, xyz[..., 2]

    # Calculate the cylindrical coordinates
    r, phi = to_polar(x, y)

    # Stack the cylindrical coordinates along the last axis
    return np.stack((r, phi, z), axis=-1)


def fourier_transform(coefficients, val):
    """
    Compute the Fourier sum of a series of coefficients.

    Args:
        coefficients (ArrayLike): The coefficients of the series.
        val (float): The value at which to evaluate the Fourier transform.

    Returns:
        float: The value of the Fourier transform at `val`.
    """
    # Calculate the angles at which to evaluate the Fourier transform
    angle = val * (np.arange(coefficients.shape[0]) + 1)

    # Evaluate the Fourier transform using the cosine and sine coefficients arrays
    return np.einsum(
        "ab,ab",  # perform a matrix multiplication between the coefficients and the cos/sin arrays
        coefficients,
        # stack the cosine and sine arrays
        np.stack((np.cos(angle), np.sin(angle)), axis=1),
    )


def unwrap_u(radius_function, dr_dphi_function, phi, num_points=100):
    """
    Calculate the length of a curve in polar coordinates.

    Args:
        radius_function (callable): Function that returns the radius at a given angle.
        dr_dphi_function (callable): Function that returns the derivative of the radius at a given angle.
        phi (float): The final angle of the curve.
        num_points (int, optional): Number of points to use for numerical integration. Defaults to 100.

    Returns:
        float: The length of the curve in polar coodrinates.
    """
    # Generate an array of angles from 0 to phi
    all_phi = np.linspace(0, phi, num_points)

    # Calculate radii and their derivatives at all points
    all_dr_dphi = np.array([dr_dphi_function(p) for p in all_phi])
    all_radii = np.array([radius_function(p) for p in all_phi])

    # Calculate the integrand as the square root of the sum of the squares of the derivatives and radii
    integrand = np.sqrt(all_dr_dphi**2 + all_radii**2)

    # Numerically integrate the integrand to find the length of the curve
    return np.trapz(integrand, all_phi)


def fourier_transform_derivative(coefficients, val):
    """
    Calculate the derivative of a function using its Fourier transform.

    Args:
        coefficients (numpy.ndarray): The Fourier coefficients of the function.
        val (float): The value at which to evaluate the derivative.

    Returns:
        float: The derivative of the Fourier transform of the function.
    """
    # Generate the wavenumbers
    k = (np.arange(coefficients.shape[0]) + 1)

    # Generate the angles at which to evaluate the derivative
    angle = val * k

    # Evaluate the cosine and sine terms of the derivative
    cos_terms = np.cos(angle)
    sin_terms = np.sin(angle)

    # Calculate the derivatives of the cosine and sine terms
    d_cos = -k * sin_terms
    d_sin = k * cos_terms

    # Perform the matrix multiplication to calculate the derivative of the Fourier transform
    derivative = np.einsum(
        "ab,ab",  # perform a matrix multiplication between the coefficients and the derivatives
        coefficients,
        # stack the cosine and sine derivative arrays
        np.stack((d_cos, d_sin), axis=1),
    )

    return derivative


def fourier_coefficients(li, lf, n, f):
    """
    Calculate the Fourier coefficients of a function.

    Args:
        li (float): Lower limit of the period interval.
        lf (float): Upper limit of the period interval.
        n (int): Number of coefficients to calculate.
        f (function): The function to calculate the Fourier coefficients of.

    Returns:
        tuple: A tuple containing the constant term and the coefficients of the function.
              The constant term is the mean value of the function over the interval.
              The coefficients are a numpy array of shape (n, 2), where each row represents the
              coefficients of a cosine and sine term.
    """
    # Calculate the interval half period
    l = (lf - li) / 2

    # Calculate the constant term
    # The constant term is twice the mean value of the function over the interval
    a0 = 1 / l * quad(lambda x: f(x), li, lf)[0]

    # Initialize arrays to store the cosine and sine coefficients
    A = np.zeros((n))
    B = np.zeros((n))

    # Calculate the coefficients of the cosine and sine terms
    coefs = []
    for i in range(1, n + 1):
        # Calculate the angular frequency
        omega = i * np.pi / l

        # Calculate the cosine coefficient
        A = quad(lambda x: f(x) * np.cos(omega * x), li, lf)[0] / l

        # Calculate the sine coefficient
        B = quad(lambda x: f(x) * np.sin(omega * x), li, lf)[0] / l

        # Append the coefficients as a numpy array to the list of coefficients
        coefs.append(np.array([A, B]))

    # Convert the list of coefficients to a numpy array
    coefs = np.stack(coefs, axis=0)

    # Return the constant term and the coefficients
    return a0 / 2.0, coefs


def get_min_dist(S1, S2):
    """
    Compute the minimum distance between two sets of points.

    Parameters
    ----------
    S1, S2 : numpy.ndarray
        Arrays of shape (N, 3) representing the points.

    Returns
    -------
    float
        The minimum distance between the points.
    """
    # Reshape the arrays to have shape (N, 3)
    S1 = np.reshape(S1, (-1, 3))
    S2 = np.reshape(S2, (-1, 3))

    # Compute the pairwise distances between points in S1 and S2
    dists = cdist(S1, S2)

    # Return the minimum distance
    return dists.min()

    # Alternative implementation using jax.numpy for differentiability
    # return np.linalg.norm(S1.P[...,None,None,:]-S2.P[None,None,...], axis=-1).min()


def fit_to_surface(fitted_surface, surface, distance: float = 0.0):
    """
    Tries to find approximately the smallest `fitted_surface` enclosing `surface`
    assuming `surface` has get_major_radius and get_minor_radius methods

    Args:
        fitted_surface: The surface to fit.
        surface: The surface to be fitted to.
        distance: The distance between the two surfaces.

    Returns:
        The fitted surface.
    """

    # Get the major and minor radii of the surface
    major_radius = surface.get_major_radius()
    minor_radius = surface.get_minor_radius(vmec=False)

    # Create a copy of the fitted surface
    new_surf = fitted_surface.model_copy()
    radius = minor_radius + distance

    # Update the parameters of the fitted surface
    # The radius is set to twice the minor radius
    # The distance to the origin is set to the major radius
    new_surf.surface_factories[0].update_params(
        radius=radius,
        distance=major_radius,
    )

    # Find the minimum distance between the fitted surface and the surface
    min_dist = new_surf().get_min_distance(surface.xyz)

    while min_dist < minor_radius/10:
        # Update the radius of the fitted surface
        radius += min_dist
        new_surf.surface_factories[0].update_params(
            radius=radius,
        )
        # Find the minimum distance between the fitted surface and the surface
        min_dist = new_surf().get_min_distance(surface.xyz)
    return new_surf


def get_principles(hess_xyz, jac_xyz, normal_unit):
    """
    Compute the principal curvatures and directions.

    Args:
        hess_xyz (ArrayLike): The Hessian matrix of the surface at each point.
        jac_xyz (ArrayLike): The Jacobian matrix of the surface at each point.
        normal_unit (ArrayLike): The unit normal vector of the surface at each point.

    Returns:
        Tuple[ArrayLike, ArrayLike]: The maximum and minimum principal curvatures.
    """

    # Compute the second partial derivatives of the surface
    dpsi_uu = hess_xyz[..., 0, 0]
    dpsi_uv = hess_xyz[..., 1, 1]
    dpsi_vv = hess_xyz[..., 0, 1]

    # Compute the first fundamental form of the surface
    # I = E*d2u + 2*F*dudv + G*d2v
    E = np.einsum("ijl,ijl->ij", jac_xyz[..., 0], jac_xyz[..., 0])
    F = np.einsum("ijl,ijl->ij", jac_xyz[..., 0], jac_xyz[..., 1])
    G = np.einsum("ijl,ijl->ij", jac_xyz[..., 1], jac_xyz[..., 1])

    # Compute the second fundamental form of the surface
    # II = L*d2u + 2*M*dudv + N*d2v
    L = np.einsum("ijl,ijl->ij", dpsi_uu, normal_unit)  # e
    M = np.einsum("ijl,ijl->ij", dpsi_uv, normal_unit)  # f
    N = np.einsum("ijl,ijl->ij", dpsi_vv, normal_unit)  # g

    # Compute the Gaussian and mean curvatures
    # Gaussian curvature K = Pmax * Pmin
    K = (L * N - M**2) / (E * G - F**2)
    # Mean curvature H = (Pmax + Pmin) / 2
    H = (L*G + E*N - 2 * F * M) / (2*(E * G - F**2))

    # Compute the principal curvatures
    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)

    return Pmax, Pmin


def get_net_current(coil_surf, toroidal: bool = True):
    """
    Compute the net toroidal or poloidal current.

    Args:
        coil_surf (CoilSurface): The coil surface.
        toroidal (bool, optional): If True, compute the net current along the toroidal axis.
            Otherwise, compute the net current along the poloidal axis. Defaults to True.

    Returns:
        ArrayLike: The net current along the specified axis.
    """

    # Get the 3D current density and Jacobian matrix
    j_3d = coil_surf.j_3d
    jac_xyz = coil_surf.jac_xyz

    # Choose the integration vector and corresponding length
    if toroidal:
        int_vec = jac_xyz[..., 0]  # u
        dl = coil_surf.du
        axis_sum = 0  # Sum over toroidal index
    else:
        int_vec = jac_xyz[..., 1]  # v
        dl = coil_surf.dv
        axis_sum = 1  # Sum over poloidal index

    # Compute the unit vector perpendicular to the integration vector and to the surface normal
    vecn = np.cross(int_vec, coil_surf.normal, -1, -1, -1)
    vecn = vecn / np.linalg.norm(vecn, axis=-1, keepdims=True)

    # Compute the projection of the 3D current density onto the integration vector
    j_tor = np.einsum("ija,ija->ij", j_3d, vecn)

    # Compute the net current along the chosen axis
    net_current = np.sum(j_tor * np.linalg.norm(int_vec,
                         axis=-1), axis=axis_sum) * dl

    return net_current
