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
    Compute the Laplace force of a distribution of
    currents on itself.

    Args:
        * j_3d: surface current Nu x Nv x 3
        * xyz: surface cartesian coordinates Nu x Nv x 3
        * normal_unit: surface normal vector normalized Nu x Nv x 3
        * ds: surface area of each sample point Nu x Nv
        * g_up_map: map from cartisian to contravariant coordinate Nu x Nv x 3 x 2
        * du: length between two naighbor points on the surface along the poloidal dimension
        * dv: length between two naighbor points on the surface along the toroidal dimension
        * end_u: cut the points along u at end_u (this parameter is there for checking the coherence with older implementations)
        * end_v: cut the points along v at end_v

    y is in the first two dimensions: ij
    x is in the next two dimensions: kl
    a,b,c are dimensions of 3d vectors (always equal to 3)
    t,u are dimensions in the 2d surface contravariant or covariant space (always equal to 2)

    _b indicate a variable belonging to the surface generating the magnetic field
    _b indicate a variable belonging to the surface where the force is computed
    """

    # norm = normal_unit
    j1 = j_3d_f
    j2 = j_3d_b[:end_u, :end_v]
    norm2_b = normal_unit_b[:end_u, :end_v]
    ds_int = ds_b[:end_u, :end_v]

    # Kernels
    T = (xyz_f[:, :, None, None] - xyz_b[None, None, ...])[:, :, :end_u, :end_v]
    inv_l1 = 1 / np.linalg.norm(T, axis=-1)
    inv_l1 = np.where(np.abs(inv_l1) > 1e10, 0.0, inv_l1)
    K = T * inv_l1[..., None] ** 3

    # Projects j_3d on the surface
    pi_x_b = np.eye(3)[None, None] - normal_unit_b[..., None] * normal_unit_b[..., None, :]
    pi_x_j1 = np.einsum("klab,ijb->ijkla", pi_x_b, j1)

    # Maps j_3d to the contravariant surface 2d space
    pi_x_j1_uv = np.einsum("klab,ijkla->ijklb", g_up_map_b, pi_x_j1)

    # get the surface divergence
    div_pix_j1 = surf_div(pi_x_j1_uv, ds=ds_b[None, None], du=du, dv=dv, axes=(2, 3))

    # Surface gradients of j
    # if grad_j_3d is None:
    grad_j_3d_b = np.stack(np.gradient(j_3d_b, du, dv, axis=(0, 1)), axis=-1)

    # Maps the surface gradients of j from cartesian to contravariant
    grad_j_contr_map = np.einsum("klat,klbt->klab", grad_j_3d_b, g_up_map_b)

    # Gets the divergence of pi_x
    pi_x_uv = np.einsum("klat,klab,bc->klct", g_up_map_b, pi_x_b, np.eye(3))
    div_pi_x_b = surf_div(pi_x_uv, du=du, dv=dv, ds=ds_b[..., None])

    # -1/(y-x) *(div pi_x j_1(y))*j_2(x)
    fac1 = -np.einsum("ijkl,ijkl,kla,kl->ija", inv_l1, div_pix_j1[:, :, :end_u, :end_v], j2, ds_int)

    # -1/(y-x) *(pi_x j_1(y) \dot \nabla) j_2(x)
    fac2 = -np.einsum(
        "ijkl,ijklt,klat,kl->ija", inv_l1, pi_x_j1_uv[:, :, :end_u, :end_v, :], grad_j_3d_b[:end_u, :end_v], ds_int
    )

    # 1/(y-x) \nabla_x <j1(y) j2(x) >
    fac3 = np.einsum("ijkl,ija,klab,kl->ijb", inv_l1, j1, grad_j_contr_map[:end_u, :end_v], ds_int)

    # <j1(y) n(x) > <y-x,n(x)> /|y-x|^3 j2(x)
    fac4 = np.einsum(
        "ija,kla,ijklc,klc,klb,kl->ijb", j1, norm2_b, K, norm2_b, j2, ds_int
    )

    # -<j1(y) j2(x)> <y−x,n(x)>/|y-x|^3 n(x)
    fac5 = -np.einsum(
        "ijklb,ija,kla,klb,klc,kl->ijc", K, j1, j2, norm2_b, norm2_b, ds_int
    )

    # 1/(y-x) <j1(y) j2(x) > div \pi_x
    fac6 = np.einsum("ijkl,ija,kla,klb,kl->ijb", inv_l1, j1, j2, div_pi_x_b[:end_u, :end_v], ds_int)

    lap_force = 1e-7 * (fac1 + fac2 + fac3 + fac4 + fac5 + fac6) * du * dv

    return lap_force


def surf_div(ten, ds, du, dv, axes=(0, 1)):
    """
    Surface divergence, where ten is an Nu x Nv x 2 tensor in contravariant
    coordinate (or Nu x Nv x Nc x 2)
    """
    du_surf_ten = np.gradient(ds * ten[..., 0], du, axis=axes[0])
    dv_surf_ten = np.gradient(ds * ten[..., 1], dv, axis=axes[1])

    return (du_surf_ten + dv_surf_ten) / ds
