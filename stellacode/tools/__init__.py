import numpy as onp
from opt_einsum import contract
from scipy.constants import mu_0

from stellacode import np

# the completely antisymetric tensor
eijk = onp.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
eijk = np.asarray(eijk)


def get_tensor_distance(S1, S2, rot_tensor):
    """S1 has a grid lu1 x lv1 and S2 lu2 x lv2x3 with the same Np
    return the tensor Npxlu1xlv1xlu2xlv2 x 3 of the vector between
    such that T[i,j,k,l,m,n] is the nth coordinates bwn the rotation of
    2pi i/Np psi1_j,k and psi2_l,m"""
    return (
        S2.P[np.newaxis, np.newaxis, np.newaxis, :, :, :]
        - np.einsum("opq,ijq->oijp", rot_tensor, S1.P)[:, :, :, np.newaxis, np.newaxis, :]
    )


def phi_coeff_from_nb(k, phisize):
    """the coefficents of Phi are store as 1d array,
    this function make the conversion with a 2d matrix"""
    lm, ln = phisize
    if k < ln:
        return (0, k + 1)
    else:
        kk = k - ln
        return (1 + kk // (2 * ln + 1), kk % (2 * ln + 1) - ln)


def get_matrix_dPhi(phisize, grids):
    """generate the tensor (2+lc) xluxlvx2 of divergence free vector fields
    lc is the number of fourier component given by phisize"""
    lm, ln = phisize
    ugrid, vgrid = grids
    lu, lv = ugrid.shape
    lc = lm * (2 * ln + 1) + ln
    matrix_dPhi = np.zeros((lc + 2, lu, lv, 2))
    # we start with du:
    matrix_dPhi = matrix_dPhi.at[0, :, :, 0].set(np.ones((lu, lv)))
    # dv
    matrix_dPhi = matrix_dPhi.at[1, :, :, 1].set(np.ones((lu, lv)))
    for coeff in range(lc):
        m, n = phi_coeff_from_nb(coeff, phisize)
        # Phi=sin(2pi(mu+nv))
        # \nabla\perp Phi = (-2 pi n cos(2pi(mu+nv)),2pi m cos(2pi(mu+nv)))
        matrix_dPhi = matrix_dPhi.at[coeff + 2, :, :, 0].set(
            -2 * np.pi * n * np.cos(2 * np.pi * (m * ugrid + n * vgrid))
        )
        matrix_dPhi = matrix_dPhi.at[coeff + 2, :, :, 1].set(
            2 * np.pi * m * np.cos(2 * np.pi * (m * ugrid + n * vgrid))
        )
    return matrix_dPhi


def get_matrix_dPhinp(phisize, grids):
    """generate the tensor (2+lc) xluxlvx2 of divergence free vector fields
    lc is the number of fourier component given by phisize"""
    import numpy as np

    lm, ln = phisize
    ugrid, vgrid = grids
    lu, lv = ugrid.shape
    lc = lm * (2 * ln + 1) + ln
    matrix_dPhi = np.zeros((lc + 2, lu, lv, 2))
    # we start with du:
    matrix_dPhi[0, :, :, 0] = np.ones((lu, lv))
    # dv
    matrix_dPhi[1, :, :, 1] = np.ones((lu, lv))
    for coeff in range(lc):
        m, n = phi_coeff_from_nb(coeff, phisize)
        # Phi=sin(2pi(mu+nv))
        # \nabla\perp Phi = (-2 pi n cos(2pi(mu+nv)),2pi m cos(2pi(mu+nv)))
        matrix_dPhi[coeff + 2, :, :, 0] = -2 * np.pi * n * np.cos(2 * np.pi * (m * ugrid + n * vgrid))
        matrix_dPhi[coeff + 2, :, :, 1] = 2 * np.pi * m * np.cos(2 * np.pi * (m * ugrid + n * vgrid))

    return matrix_dPhi


def get_matrix_dPhi_2(phisize, grids):
    """New representation"""
    lm, ln = phisize
    ugrid, vgrid = grids
    lu, lv = ugrid.shape
    lc = lm * (2 * ln + 1) + ln
    matrix_dPhi = np.zeros((lc + 2, lu, lv, 2))
    # we start with du:
    matrix_dPhi[0, :, :, 0] = np.ones((lu, lv))
    # dv
    matrix_dPhi[1, :, :, 1] = np.ones((lu, lv))
    for coeff in range(lc):
        m, n = phi_coeff_from_nb(coeff, phisize)
        # Phi=sin(2pi(mu+nv))
        # \nabla\perp Phi = (-2 pi n cos(2pi(mu+nv)),2pi m cos(2pi(mu+nv)))
        matrix_dPhi[coeff + 2, :, :, 0] = -np.pi * n * np.cos(2 * np.pi * m * ugrid) * np.cos(np.pi * n * vgrid)
        matrix_dPhi[coeff + 2, :, :, 1] = -2 * np.pi * m * np.sin(2 * np.pi * m * ugrid) * np.sin(np.pi * n * vgrid)
    return matrix_dPhi


def get_matrix_dPhi_cylinders(phisize, grids, ncyl):
    lm, ln = phisize
    ugrid, vgrid = grids
    lu, lv = ugrid.shape
    lc = lm * (2 * ln + 1) + ln
    matrix_dPhi = np.zeros((lc * ncyl + 2, lu, lv, 2))
    # we start with du:
    matrix_dPhi[0, :, :, 0] = np.ones((lu, lv))
    # dv
    matrix_dPhi[1, :, :, 1] = np.ones((lu, lv))
    if ncyl != 3:
        lv_cyl = lv // ncyl  # number of toroidal points per cylinder
        for i in range(ncyl):
            im = lc * i + 2
            # v index corresponding to where the cylinder we are in starts and ends
            vm, vM = i * lv_cyl, (i + 1) * lv_cyl
            vmax = 1 / ncyl  # scaling factor, to have 0 at both junctions
            for coeff in range(lc):
                m, n = phi_coeff_from_nb(coeff, phisize)
                matrix_dPhi[im + coeff, :, vm:vM:, 0] = (
                    -np.pi
                    * n
                    * np.cos(2 * np.pi * m * ugrid[:, :lv_cyl:])
                    * np.cos(np.pi * n * vgrid[:, :lv_cyl:] / vmax)
                    / vmax
                )
                matrix_dPhi[im + coeff, :, vm:vM:, 1] = (
                    -2
                    * np.pi
                    * m
                    * np.sin(2 * np.pi * m * ugrid[:, :lv_cyl:])
                    * np.sin(np.pi * n * vgrid[:, :lv_cyl:] / vmax)
                )
    return matrix_dPhi


def get_rot_tensor(Np):
    rot = np.array(
        [
            [np.cos(2 * np.pi / Np), -np.sin(2 * np.pi / Np), 0],
            [np.sin(2 * np.pi / Np), np.cos(2 * np.pi / Np), 0],
            [0, 0, 1],
        ]
    )
    return np.stack([np.linalg.matrix_power(rot, i) for i in range(Np)])


def compute_Qj(matrixd_phi, dpsi, dS):
    """take only the segment whitout rotation of j"""
    lu, lv = dS.shape
    Qj = np.einsum(
        "oija,ijda,ijdk,pijk,ij->op",
        matrixd_phi,
        dpsi,
        dpsi,
        matrixd_phi,
        1 / dS,
        optimize=True,
    ) / (lu * lv)
    return Qj
