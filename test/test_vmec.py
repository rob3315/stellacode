import pytest


@pytest.mark.skip("Missing dependency")
def test_vmec():
    import numpy as np
    import utilitiesRF as urf

    from stellacode.tools.vmec import VMECIO

    vmec = VMECIO("data/w7x/wout_d23p4_tm.nc")
    vmec2 = urf.VmecIO()
    vmec2.read_wout("data/w7x/wout_d23p4_tm.nc")
    vmec2.fields(nradius=200, ntheta=10, nzeta=10)
    import pdb;pdb.set_trace()

    bvec = np.stack((vmec2.Bx, vmec2.By, vmec2.BZ), axis=-1)
    bnorm = np.linalg.norm(bvec, axis=-1)
    np.sqrt(vmec2.Bx**2+ vmec2.By**2+ vmec2.BZ**2)

    vmec2.B

    theta = np.linspace(0, 2 * np.pi, num=10)
    zeta = np.linspace(0, 2 * np.pi, num=10)
    zeta2D, theta2D = np.meshgrid(zeta, theta)

    irad0 = np.linspace(0, 200, num=200, endpoint=True).round()
    irad = [int(i) for i in irad0]

    bvmec = vmec.get_b_vmec(zeta2D, theta2D, surf_labels=irad)
    assert np.abs(np.transpose(bvmec[..., 0], (2, 1, 0)) - vmec2.B_s).max() < 1e-5
    assert np.abs(np.transpose(bvmec[..., 1], (2, 1, 0)) - vmec2.B_u).max() < 1e-6
    assert np.abs(np.transpose(bvmec[..., 2], (2, 1, 0)) - vmec2.B_v).max() < 5e-5

    b_cart = vmec.get_b_cartesian(zeta2D, theta2D, surf_labels=irad)
    b_val = vmec.get_val("b", zeta2D, theta2D)
