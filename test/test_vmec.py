import pytest
from stellacode.tools.vmec import VMECIO
import numpy as np


# @pytest.mark.skip("Missing dependency")
def test_vmec():
    import utilitiesRF as urf
    ntheta = 4
    nzeta = 4

    vmec = VMECIO("data/w7x/wout_d23p4_tm.nc", ntheta=ntheta, nzeta=nzeta)
    vmec2 = urf.VmecIO()
    vmec2.read_wout("data/w7x/wout_d23p4_tm.nc")


    vmec2.flux_surf(nradius=vmec2.ns, ntheta=ntheta, nzeta=nzeta)
    vmec2.fields(nradius=vmec2.ns, ntheta=ntheta, nzeta=nzeta)

    # grid = vmec.get_grid(ntheta, nzeta)

    pos = vmec.rphiz
    assert np.allclose(vmec2.R, pos[..., 0])
    assert np.allclose(vmec2.Z, pos[..., -1])

    bvmec = vmec.get_b_vmec()
    bvmec_gt = np.stack((vmec2.B_u, vmec2.B_v, vmec2.B_s), axis=-1)
    assert np.allclose(bvmec_gt, bvmec)

    R_grad = vmec.get_val_grad("r", nyq=False)
    R_grad_gt = np.stack((vmec2.drdu, vmec2.drdv), axis=-1)
    assert np.allclose(R_grad, R_grad_gt)

    Z_grad = vmec.get_val_grad("z", nyq=False)
    Z_grad_gt = np.stack((vmec2.dzdu, vmec2.dzdv), axis=-1)
    assert np.allclose(Z_grad, Z_grad_gt)

    bcyl = vmec.b_cylindrical
    bcyl_gt = np.stack((vmec2.BR, vmec2.Bphi, vmec2.BZ), axis=-1)
    assert np.allclose(bcyl_gt, bcyl)

    b_cart = vmec.b_cartesian
    b_cart_gt = np.stack((vmec2.Bx, vmec2.By, vmec2.BZ), axis=-1)
    assert np.allclose(b_cart_gt, b_cart)

    # jvmec = vmec.j_vmec
    # jvmec_gt = np.stack((vmec2.Ju, vmec2.Jv), axis=-1)
    # assert np.allclose(jvmec_gt, jvmec)