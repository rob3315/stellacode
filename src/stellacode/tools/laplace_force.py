from jax import Array

from stellacode import np


def laplace_force(
    j_3d_f: Array,
    xyz_f: Array,
    j_3d_b: Array,
    xyz_b: Array,
    normal_unit_b: Array,
    ds_b: Array,
    g_up_map_b: Array,
    du: float,
    dv: float,
    end_u: int = 1000000,
    end_v: int = 1000000,
) -> Array:
    """
    Compute the Laplace force of a distribution of currents on itself.

    Args:
        * j_3d_f: surface current Nu x Nv x 3
        * xyz_f: surface cartesian coordinates Nu x Nv x 3
        * j_3d_b: surface current Nu x Nv x 3
        * xyz_b: surface cartesian coordinates Nu x Nv x 3
        * normal_unit_b: surface normal vector normalized Nu x Nv x 3
        * ds_b: surface area of each sample point Nu x Nv
        * g_up_map_b: map from cartisian to contravariant coordinate Nu x Nv x 3 x 2
        * du: length between two neighbor points on the surface along the poloidal dimension
        * dv: length between two neighbor points on the surface along the toroidal dimension
        * end_u: cut the points along u at end_u (this parameter is there for checking the coherence with older implementations)
        * end_v: cut the points along v at end_v

    Returns:
        * array of shape (3,) containing the force along the x, y, z axes.

    The Laplace force is computed as:
    -1/(y-x) *(div pi_x j_1(y))*j_2(x)
    -1/(y-x) *(pi_x j_1(y) \dot \nabla) j_2(x)
    1/(y-x) \nabla_x <j1(y) j2(x) >
    <j1(y) n(x) > <y-x,n(x)> /|y-x|^3 j2(x)
    -<j1(y) j2(x)> <y−x,n(x)>/|y-x|^3 n(x)
    1/(y-x) <j1(y) j2(x) > div \pi_x

    """

    # Define the variables
    j1 = j_3d_f  # surface current at point y
    j2 = j_3d_b[:end_u, :end_v]  # surface current at point x
    # surface normal vector normalized at point x
    norm2_b = normal_unit_b[:end_u, :end_v]
    ds_int = ds_b[:end_u, :end_v]  # surface area at point x

    # Compute the kernels
    T = (xyz_f[:, :, None, None] -
         xyz_b[None, None, ...])[:, :, :end_u, :end_v]
    inv_l1 = 1 / np.linalg.norm(T, axis=-1)
    inv_l1 = np.where(np.abs(inv_l1) > 1e10, 0.0, inv_l1)
    K = T * inv_l1[..., None] ** 3

    # Projects j_3d on the surface
    pi_x_b = np.eye(3)[None, None] - normal_unit_b[...,
                                                   None] * normal_unit_b[..., None, :]
    pi_x_j1 = np.einsum("klab,ijb->ijkla", pi_x_b, j1)

    # Maps j_3d to the contravariant surface 2d space
    pi_x_j1_uv = np.einsum("klab,ijkla->ijklb", g_up_map_b, pi_x_j1)

    # get the surface divergence
    div_pix_j1 = surf_div(
        pi_x_j1_uv, ds=ds_b[None, None], du=du, dv=dv, axes=(2, 3))

    # Surface gradients of j
    grad_j_3d_b = np.stack(np.gradient(j_3d_b, du, dv, axis=(0, 1)), axis=-1)

    # Maps the surface gradients of j from cartesian to contravariant
    grad_j_contr_map = np.einsum("klat,klbt->klab", grad_j_3d_b, g_up_map_b)

    # Gets the divergence of pi_x
    pi_x_uv = np.einsum("klat,klab,bc->klct", g_up_map_b, pi_x_b, np.eye(3))
    div_pi_x_b = surf_div(pi_x_uv, du=du, dv=dv, ds=ds_b[..., None])

    # Compute the Laplace force
    fac1 = -np.einsum("ijkl,ijkl,kla,kl->ija", inv_l1,
                      div_pix_j1[:, :, :end_u, :end_v], j2, ds_int)
    fac2 = -np.einsum(
        "ijkl,ijklt,klat,kl->ija", inv_l1, pi_x_j1_uv[:, :,
                                                      :end_u, :end_v, :], grad_j_3d_b[:end_u, :end_v], ds_int
    )
    fac3 = np.einsum("ijkl,ija,klab,kl->ijb", inv_l1, j1,
                     grad_j_contr_map[:end_u, :end_v], ds_int)
    fac4 = np.einsum(
        "ija,kla,ijklc,klc,klb,kl->ijb", j1, norm2_b, K, norm2_b, j2, ds_int
    )
    fac5 = -np.einsum(
        "ijklb,ija,kla,klb,klc,kl->ijc", K, j1, j2, norm2_b, norm2_b, ds_int
    )
    fac6 = np.einsum("ijkl,ija,kla,klb,kl->ijb", inv_l1, j1,
                     j2, div_pi_x_b[:end_u, :end_v], ds_int)

    lap_force = 1e-7 * (fac1 + fac2 + fac3 + fac4 + fac5 + fac6) * du * dv

    return lap_force


def surf_div(ten, ds, du, dv, axes=(0, 1)):
    """
    Compute the surface divergence of a tensor field.

    Parameters
    ----------
    ten : ndarray
        Tensor field on a surface.
    ds : ndarray
        Surface element.
    du : float
        Grid spacing in the u direction.
    dv : float
        Grid spacing in the v direction.
    axes : tuple, optional
        Axes of the tensor field. Default is (0, 1).

    Returns
    -------
    ndarray
        Surface divergence of the tensor field.

    Notes
    -----
    The tensor field is assumed to be in contravariant coordinate, i.e.,
    the first and second indices are the u and v coordinates,
    respectively.
    """
    # Compute the surface divergence of the first and second indices
    # of the tensor field.
    du_surf_ten = np.gradient(ds * ten[..., 0], du, axis=axes[0])
    dv_surf_ten = np.gradient(ds * ten[..., 1], dv, axis=axes[1])

    # Compute the surface divergence by summing the divergences of the
    # first and second indices and dividing by the surface element.
    return (du_surf_ten + dv_surf_ten) / ds
