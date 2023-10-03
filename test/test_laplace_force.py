import configparser

import jax

jax.config.update("jax_enable_x64", True)
import pytest
from scipy.io import netcdf_file

from stellacode import np
from stellacode.costs.em_cost import EMCost, get_b_field_err
from stellacode.definitions import w7x_plasma, ncsx_plasma
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface.imports import get_net_current
from stellacode.surface import CylindricalSurface, rotate_coil, FourierSurface
from stellacode.surface.factories import get_original_cws
from stellacode.tools.laplace_force import laplace_force
from stellacode.surface.factory_tools import Sequential


def test_laplace_force():
    # major_radius = 5.5
    # minor_radius = 1.404687741189692  # 0.9364584941264614*(1+0.5)
    lu, lv = 11 + 1, 13 + 1

    # coil_surf = get_original_cws(path_cws=ncsx_plasma.path_cws,
    #                              path_plasma=ncsx_plasma.path_plasma, n_harmonics = 2, factor = 4)

    fourier_factory = FourierSurface.from_file(
        ncsx_plasma.path_cws,
        integration_par=IntegrationParams(num_points_u=lu, num_points_v=lv),
        n_fp=5,
    )
    I, G = 1e6, 2e5
    import numpy as onp

    onp.random.seed(987)
    factory = Sequential(
        surface_factories=[
            fourier_factory,
            rotate_coil(
                current=Current(num_pol=2, num_tor=2, cos_basis=True, net_currents=np.array([I, G])),
                nfp=fourier_factory.nfp,
            ),
        ]
    )

    # current_n_coeff = 8
    # n_points = current_n_coeff * 4
    # em_cost = EMCost.from_plasma_config(
    #     plasma_config=w7x_plasma,
    #     integration_par=IntegrationParams(num_points_u=8, num_points_v=8),
    #     lamb=1e-16,
    # )

    # plasma_surf = FourierSurface.from_file(
    #     w7x_plasma.path_plasma,
    #     integration_par=IntegrationParams(num_points_u=n_points, num_points_v=n_points),
    # )
    # cost, metrics, results = em_cost.cost(coil_surf)

    m, n = 2, 2
    l = 2 * (m * (2 * n + 1) + n)
    lst_coeff = 1e3 * (2 * onp.random.random(l) - 1) / (onp.arange(1, l + 1) ** 2)

    factory.surface_factories[1].surface_factories[0].current.phi_mn = lst_coeff / 1e8  # because of the scaling, a setter would be better.
    coil_surf = factory()
    force = coil_surf.naive_laplace_force(
        epsilon=np.min(np.linalg.norm(coil_surf.xyz[1:] - coil_surf.xyz[:-1], axis=-1)) * 2
    )

    np.max(np.linalg.norm(force, axis=-1))

    force2 = laplace_force(
        j_3d=coil_surf.get_j_3D(),
        xyz=coil_surf.xyz,
        normal_unit=coil_surf.normal_unit,
        ds=coil_surf.ds,
        g_up_map=coil_surf.get_g_upper_basis(),
        nfp=fourier_factory.nfp,
        du=1 / 11,
        dv=1 / 14,
        end_u=-1,
        end_v=14,
    )

    # Check that the code for new and old versions of Laplace force are the same
    force3 = f_laplace(coil_surf, 0, 1, Np=1, lu=12, lv=14)
    assert np.allclose(force2[0, 1], force3)

    force3 = f_laplace(coil_surf, 5, 10, Np=1, lu=12, lv=14)
    assert np.allclose(force2[5, 10], force3)

    force2 = coil_surf.laplace_force(nfp=fourier_factory.nfp)

    # Approximate and rigorous Laplace forces are close:
    assert np.max(np.linalg.norm(force[:, :14] - force2, axis=-1)) / np.mean(np.linalg.norm(force2)) < 0.1

    # The Laplace Force is pointing outside of the surface
    assert np.mean(np.einsum("ija,ija->ij", force2, coil_surf.normal_unit[:, :14])) < -1e4


def div(f, ds):
    """return the divergence of f : lu x lv x 2"""
    import numpy as np

    res = np.zeros((f.shape[0], f.shape[1]))
    lu, lv = f.shape[:2]
    for i in range(2):  # both coefficients
        aux = ds * f[:, :, i]
        d1, d2 = (1 / (lu - 1), 1 / (lv - 1))
        daux = np.gradient(aux, d1, d2)[i]
        # periodize_function(daux)
        res += daux / ds
    return res


def vector_field_pull_back(f, jac_xyz):
    """inverse of the push forward for vector field, f has to be already tangent to S"""
    # still not optimal
    # TODO improve
    import numpy as np

    lu, lv = jac_xyz.shape[:2]
    X = np.zeros((lu, lv, 2))
    # dpsi=np.array([self.surf.dpsidu,self.surf.dpsidv])
    dpsi = np.transpose(jac_xyz, (3, 2, 0, 1))

    A = np.einsum("ji...,ki...->jk...", dpsi, dpsi)
    b = np.einsum("ij...,...j->i...", dpsi, f)
    AA = np.moveaxis(A, [0, 1, 2, 3], [2, 3, 0, 1])
    bb = np.moveaxis(b, [0, 1, 2], [2, 0, 1])
    X = np.linalg.solve(AA, bb)
    return X


def pi_x(f, norm):
    import numpy as np

    """return the projection of the vector field f : lu x lv x 3 on the tangent bundle of self.surf"""
    return f - np.repeat(np.sum(f * norm, axis=2)[:, :, np.newaxis], 3, axis=2) * norm


def f_laplace(surf, i, j, Np, lu, lv):
    import numpy as np

    res1 = np.zeros(3)
    res2 = np.zeros(3)
    res3 = np.zeros(3)
    res4 = np.zeros(3)
    res5 = np.zeros(3)
    res6 = np.zeros(3)
    res = np.zeros(3)
    # Np = surf.nfp

    lv = lv + 1  # because the code expects the points at the edges
    # of the surface are duplicated to ensure precise gradient calculation at the edges.
    du, dv = (1 / (lu - 1), 1 / (lv - 1))
    X_ = np.array(surf.xyz[:, :lv, 0])
    Y_ = np.array(surf.xyz[:, :lv, 1])
    Z_ = np.array(surf.xyz[:, :lv, 2])
    norm = surf.normal_unit[:, :lv, :]
    jac_xyz = surf.jac_xyz[:, :lv, :, :]
    ds = surf.ds[:, :lv]
    g_upper = surf.get_g_upper_contravariant()[:, :lv]

    j2 = np.array(surf.get_j_3D())[:, :lv, :]

    j1 = j2[i, j, :]
    rot = np.array(
        [
            [np.cos(2 * np.pi / Np), -np.sin(2 * np.pi / Np), 0],
            [np.sin(2 * np.pi / Np), np.cos(2 * np.pi / Np), 0],
            [0, 0, 1],
        ]
    )
    lst_rot = np.array([np.linalg.matrix_power(rot, l) for l in range(Np + 1)])
    for l in range(Np):
        # we rotate j1 of rot^l
        rot_l = lst_rot[l]
        j1y = np.matmul(lst_rot[l], j1)
        j1ya = np.tile(j1y, (lu, lv, 1))  # lu x lv x 3
        newY = np.dot(rot_l, np.array([X_[i, j], Y_[i, j], Z_[i, j]]))
        d2 = (newY[0] - X_) ** 2 + (newY[1] - Y_) ** 2 + (newY[2] - Z_) ** 2
        if l == 0:
            d2[i, j] = 1  # for the division
        elif j == 0 and l == 1:
            d2[i, -1] = 1
            if i == 0:
                d2[-1, -1] = 1
        iymx = 1 / np.sqrt(d2)
        if l == 0:
            iymx[i, j] = 0  # by convention
        pi_xjy = pi_x(j1ya, norm=norm)  # lu x lv x 3
        pi_xjy_uv = vector_field_pull_back(pi_xjy, jac_xyz=jac_xyz)  # lu x lv x 2
        # -1/(y-x) *(div pi_x j_1(y))*j_2(x)

        # j1ya -
        # np.repeat(np.sum(j1ya * norm, axis=2)[:, :, np.newaxis], 3, axis=2) * norm

        # pi_x2 = np.eye(3)[None, None] - norm[..., None] * norm[..., None, :]
        # pi_x_j = np.einsum("klab,b->kla", pi_x2, j1)

        # np.einsum("ija,ija->ij", pi_xjy, norm)
        lu, lv = pi_xjy_uv.shape[:2]

        P1 = -1 * (iymx * div(pi_xjy_uv, ds=ds))[:, :, np.newaxis] * j2
        # print("P1")
        # -1/(y-x) *(pi_x j_1(y) \dot \nabla) j_2(x)
        #### Not good keep get_gradj
        # dj2=surf.get_gradj(coeff2)
        dj2 = np.transpose(np.stack(np.gradient(j2, du, dv, axis=(0, 1)), axis=-2), (2, 0, 1, 3))

        dj2x, dj2y, dj2z = dj2[:, :, :, 0], dj2[:, :, :, 1], dj2[:, :, :, 2]
        P2 = np.zeros((lu, lv, 3))
        for k in range(2):
            P2[:, :, 0] += pi_xjy_uv[:, :, k] * dj2x[k]
            P2[:, :, 1] += pi_xjy_uv[:, :, k] * dj2y[k]
            P2[:, :, 2] += pi_xjy_uv[:, :, k] * dj2z[k]
        P2 *= -1 * iymx[:, :, np.newaxis]
        # print("P2")
        # 1/(y-x) \nabla <j1(y) j2(x) >
        f = np.sum(j1ya * j2, axis=2)
        # error is here !!!!!
        # df=0*np.array(np.gradient(f,1/(lu-1),1/(lv-1)))

        df = np.sum(j1ya[np.newaxis] * dj2, axis=3)
        # df=
        # dj2du,dj2dv=
        g_up = np.transpose(g_upper, (2, 3, 0, 1))
        gradf = np.einsum("ij...,j...->i...", g_up, df)
        # gradf=np.einsum('ij...,j...->i...',surf.surf.g_upper,df)
        # periodize_function(gradf[0])
        # periodize_function(gradf[1])
        P3 = np.zeros((lu, lv, 3))
        for k in range(3):
            P3[:, :, k] += iymx * (
                gradf[0, :, :] * jac_xyz[:, :, k, 0] + gradf[1, :, :] * jac_xyz[:, :, k, 1]
            )  # lu x lv x 3
            # P3[:,:,k]+=iymx*(gradf[0,:,:]*surf.surf.dpsidu[k]+gradf[1,:,:]*surf.surf.dpsidv[k])# lu x lv x 3
        # 1/(y-x) <j1(y) j2(x) > div \pi_x

        # print("P3")
        # highly not optimal
        e1 = np.tile(np.array([1, 0, 0]), (lu, lv, 1))
        e2 = np.tile(np.array([0, 1, 0]), (lu, lv, 1))
        e3 = np.tile(np.array([0, 0, 1]), (lu, lv, 1))
        pi_xe1 = pi_x(e1, norm=norm)  # lu x lv x 3
        pi_xe1_uv = vector_field_pull_back(pi_xe1, jac_xyz=jac_xyz)  # lu x lv x 2
        pi_xe2 = pi_x(e2, norm=norm)  # lu x lv x 3
        pi_xe2_uv = vector_field_pull_back(pi_xe2, jac_xyz=jac_xyz)  # lu x lv x 2
        pi_xe3 = pi_x(e3, norm=norm)  # lu x lv x 3
        pi_xe3_uv = vector_field_pull_back(pi_xe3, jac_xyz=jac_xyz)  # lu x lv x 2
        P6 = np.zeros((lu, lv, 3))
        P6[:, :, 0] = iymx * f * div(pi_xe1_uv, ds=ds)
        P6[:, :, 1] = iymx * f * div(pi_xe2_uv, ds=ds)
        P6[:, :, 2] = iymx * f * div(pi_xe3_uv, ds=ds)
        # print("P6")
        # <j1(y) n(x) > <y-x,n(x)> /|y-x|^3 j2(x)
        ymx = np.zeros((3, lu, lv))
        ymx = np.array([newY[0] - X_, newY[1] - Y_, newY[2] - Z_])  # y-x
        K = iymx**3 * np.sum(ymx * np.transpose(norm, (2, 0, 1)), axis=0)  # K=<y-x,n(x)> /|y-x|^3
        c = K * np.sum(j1ya * norm, axis=2)
        P4 = np.einsum("...,...k->...k", c, j2)
        # print("P4")
        # -<j1(y) j2(x)> <y−x,n(x)>/|y-x|^3 n(x)dx
        P5 = -1 * np.einsum("...,...k->...k", f * K, norm)
        # print("P5")
        # Integration
        # S=self.surf.dS[:-1,:-1,np.newaxis]*(P1+P2+P3+P4+P5)[:-1,:-1,:]
        S1 = ds[:-1, :-1, np.newaxis] * P1[:-1, :-1, :]
        S2 = ds[:-1, :-1, np.newaxis] * P2[:-1, :-1, :]
        S3 = ds[:-1, :-1, np.newaxis] * P3[:-1, :-1, :]
        S4 = ds[:-1, :-1, np.newaxis] * P4[:-1, :-1, :]
        S5 = ds[:-1, :-1, np.newaxis] * P5[:-1, :-1, :]
        S6 = ds[:-1, :-1, np.newaxis] * P6[:-1, :-1, :]

        res1 += np.matmul(lst_rot[Np - l], 1e-7 * np.sum(S1, axis=(0, 1)) / ((lu - 1) * (lv - 1)))  # mu0/4pi
        res2 += np.matmul(lst_rot[Np - l], 1e-7 * np.sum(S2, axis=(0, 1)) / ((lu - 1) * (lv - 1)))  # mu0/4pi
        res3 += np.matmul(lst_rot[Np - l], 1e-7 * np.sum(S3, axis=(0, 1)) / ((lu - 1) * (lv - 1)))  # mu0/4pi
        res4 += np.matmul(lst_rot[Np - l], 1e-7 * np.sum(S4, axis=(0, 1)) / ((lu - 1) * (lv - 1)))  # mu0/4pi
        res5 += np.matmul(lst_rot[Np - l], 1e-7 * np.sum(S5, axis=(0, 1)) / ((lu - 1) * (lv - 1)))  # mu0/4pi
        res6 += np.matmul(lst_rot[Np - l], 1e-7 * np.sum(S6, axis=(0, 1)) / ((lu - 1) * (lv - 1)))  # mu0/4pi

    res = res1 + res2 + res3 + res4 + res5 + res6

    return res
