import numpy as onp

from stellacode import np
from stellacode.surface import (
    CoilFactory,
    Current,
    CurrentZeroTorBC,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface.abstract_surface import get_inv_ds_grad


def test_current_grad():
    int_par = IntegrationParams(num_points_u=32, num_points_v=32, center_vgrid=True)
    tore = ToroidalSurface(integration_par=int_par, nfp=3)

    current = CurrentZeroTorBC(num_pol=2, num_tor=2, net_currents=np.array([1, 2.0]), cos_basis=True)

    grid = np.stack(int_par.get_uvgrid(), axis=0)
    phi_mn = onp.zeros(14)
    phi_mn[:2] = 1
    phi_mn[2:5] = 1e-3
    curr_op = current(grid)
    curr_op2 = super(CurrentZeroTorBC, current).__call__(grid)

    assert onp.allclose(curr_op, np.transpose(curr_op2, (2, 0, 1, 3)), atol=1e-6)
    np.transpose(curr_op2, (2, 0, 1, 3))[2, 0, 0]

    curr_op = current.get_grad_current_op(grid)
    curr_op2 = super(CurrentZeroTorBC, current).get_grad_current_op(grid)
    np.max(np.abs(curr_op - np.transpose(curr_op2, (2, 0, 1, 3, 4))), (1, 2, 3))
    assert onp.allclose(curr_op, np.transpose(curr_op2, (2, 0, 1, 3, 4)), atol=1e-6)

    current = Current(num_pol=2, num_tor=2, net_currents=np.array([1, 2.0]), cos_basis=True)

    grid = np.stack(int_par.get_uvgrid(), axis=0)
    phi_mn = onp.zeros(2 + 12 * 2)
    phi_mn[:2] = 1
    phi_mn[2:] = 1e-3
    curr_op = current(grid)
    curr_op2 = super(Current, current).__call__(grid)
    assert onp.allclose(curr_op, np.transpose(curr_op2, (2, 0, 1, 3)), atol=1e-6)

    curr_op = current.get_grad_current_op(grid)
    curr_op2 = super(Current, current).get_grad_current_op(grid)
    assert onp.allclose(curr_op, np.transpose(curr_op2, (2, 0, 1, 3, 4)), atol=1e-6)

    coil_fac = CoilFactory(current=current, compute_grad_current_op=True)
    surf = tore()
    coil_op = coil_fac(surf)
    coil_surf = coil_op.get_coil(phi_mn=phi_mn)

    grad_js_approx = np.stack(np.gradient(coil_surf.j_surface, coil_surf.du, coil_surf.dv, axis=(0, 1)), axis=-1)
    assert np.mean(np.abs(grad_js_approx - coil_surf.grad_j_surface)) / np.mean(np.abs(coil_surf.grad_j_surface)) < 0.04

    jac_approx = np.stack(np.gradient(coil_surf.xyz, coil_surf.du, coil_surf.dv, axis=(0, 1)), axis=-1)
    assert np.mean(np.abs(jac_approx - coil_surf.jac_xyz)) / np.mean(np.abs(coil_surf.jac_xyz)) < 0.04

    hess_approx = np.stack(np.gradient(coil_surf.jac_xyz, coil_surf.du, coil_surf.dv, axis=(0, 1)), axis=-1)
    assert np.mean(np.abs(hess_approx - coil_surf.hess_xyz)) / np.mean(np.abs(coil_surf.hess_xyz)) < 0.08

    ids_grad = get_inv_ds_grad(coil_surf)
    grad_ids_approx = np.stack(np.gradient(1 / coil_surf.ds, coil_surf.du, coil_surf.dv, axis=(0, 1)), axis=-1)
    assert np.mean(np.abs(ids_grad - grad_ids_approx)) / np.mean(np.abs(ids_grad)) < 2e-2

    grad_j_3d = coil_surf.grad_j_3d
    grad_j_3d_approx = np.stack(np.gradient(coil_surf.j_3d, coil_surf.du, coil_surf.dv, axis=(0, 1)), axis=-1)
    assert np.mean(np.abs(grad_j_3d - grad_j_3d_approx)) / np.mean(np.abs(grad_j_3d)) < 1e-2
