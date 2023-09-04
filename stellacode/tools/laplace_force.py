from stellacode import np


def laplace_force(j_3d, xyz, normal_unit, ds, g_up_map, 
                  num_tor_symmetry, 
                  du, dv, 
                  end_u=1000000, end_v=1000000
                  ):
    """
    Compute the Laplace force of a distribution of
    currents on itself.


    y is in the first two dimensions: ij
    x is in the next two dimensions: kl
    a,b,c are dimensions of 3d vectors (always equal to 3)
    t,u are dimensions in the 2d surface contravariant or covariant space (always equal to 2)
    """

    norm = normal_unit
    num_pts = j_3d.shape[1] // num_tor_symmetry
    j1 = j_3d[:, :num_pts]
    j2 = j_3d[:end_u, :end_v]
    norm2 = norm[:end_u, :end_v]
    ds_int = ds[:end_u, :end_v]

    # Kernels
    T = (xyz[:, :num_pts, None, None] - xyz[None, None, ...])[:, :, :end_u, :end_v]
    inv_l1 = 1 / np.linalg.norm(T, axis=-1)
    inv_l1 = np.where(np.isinf(inv_l1), 0.0, inv_l1)
    K = T * inv_l1[..., None] ** 3

    # Projects j_3d on the surface
    pi_x = np.eye(3)[None, None] - norm[..., None] * norm[..., None, :]
    pi_x_j1 = np.einsum("klab,ijb->ijkla", pi_x, j1)

    # Maps j_3d to the contravariant surface 2d space
    pi_x_j1_uv = np.einsum("klab,ijkla->ijklb", g_up_map, pi_x_j1)

    # get the surface divergence
    div_pix_j1 = surf_div(pi_x_j1_uv, ds=ds[None, None], du=du, dv=dv, axes=(2, 3))

    # Surface gradients of j
    grad_j = np.stack(np.gradient(j_3d, du, dv, axis=(0, 1)), axis=-1)

    # Maps the surface gradients of j from cartesian to contravariant
    grad_j_contr_map = np.einsum("klat,klbt->klab", grad_j, g_up_map)

    # Gets the divergence of pi_x
    pi_x_uv = np.einsum("ijat,ijab,bc->ijct", g_up_map, pi_x, np.eye(3))
    div_pi_x = surf_div(pi_x_uv, du=du, dv=dv, ds=ds[..., None])

    # -1/(y-x) *(div pi_x j_1(y))*j_2(x)
    fac1 = -np.einsum("ijkl,ijkl,kla,kl->ija", inv_l1, div_pix_j1[:, :, :end_u, :end_v], j2, ds_int)

    # -1/(y-x) *(pi_x j_1(y) \dot \nabla) j_2(x)
    fac2 = -np.einsum(
        "ijkl,ijklt,klat,kl->ija", inv_l1, pi_x_j1_uv[:, :, :end_u, :end_v, :], grad_j[:end_u, :end_v], ds_int
    )

    # 1/(y-x) \nabla_x <j1(y) j2(x) >
    fac3 = np.einsum("ijkl,ija,klab,kl->ijb", inv_l1, j1, grad_j_contr_map[:end_u, :end_v], ds_int)

    # <j1(y) n(x) > <y-x,n(x)> /|y-x|^3 j2(x)
    fac4 = np.einsum("ija,kla,ijklc,klc,klb,kl->ijb", j1, norm2, K, norm2, j2, ds_int)

    # -<j1(y) j2(x)> <y−x,n(x)>/|y-x|^3 n(x)
    fac5 = -np.einsum("ijklb,ija,kla,klb,klc,kl->ijc", K, j1, j2, norm2, norm2, ds_int)

    # 1/(y-x) <j1(y) j2(x) > div \pi_x
    fac6 = np.einsum("ijkl,ija,kla,klb,kl->ijb", inv_l1, j1, j2, div_pi_x[:end_u, :end_v], ds_int)

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
