import numpy as np
from scipy.constants import mu_0
from opt_einsum import contract
# the completely antisymetric tensor
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1


def get_tensor_distance(S1, S2, rot_tensor):
    """S1 has a grid lu1 x lv1 and S2 lu2 x lv2x3 with the same Np
    return the tensor Npxlu1xlv1xlu2xlv2 x 3 of the vector between
    such that T[i,j,k,l,m,n] is the nth coordinates bwn the rotation of
    2pi i/Np psi1_j,k and psi2_l,m"""
    return S2.P[np.newaxis, np.newaxis, np.newaxis, :, :, :]-contract('opq,ijq->oijp', rot_tensor, S1.P)[:, :, :, np.newaxis, np.newaxis, :]


def phi_coeff_from_nb(k, phisize):
    """the coefficents of Phi are store as 1d array,
    this function make the conversion with a 2d matrix"""
    lm, ln = phisize
    if k < ln:
        return (0, k+1)
    else:
        kk = k-ln
        return (1+kk//(2*ln+1), kk % (2*ln+1)-ln)


def get_matrix_dPhi(phisize, grids):
    """generate the tensor (2+lc) xluxlvx2 of divergence free vector fields
    lc is the number of fourier component given by phisize"""
    lm, ln = phisize
    ugrid, vgrid = grids
    lu, lv = ugrid.shape
    lc = (lm*(2*ln+1)+ln)
    matrix_dPhi = np.zeros((lc+2, lu, lv, 2))
    # we start with du:
    matrix_dPhi[0, :, :, 0] = np.ones((lu, lv))
    # dv
    matrix_dPhi[1, :, :, 1] = np.ones((lu, lv))
    for coeff in range(lc):
        m, n = phi_coeff_from_nb(coeff, phisize)
        # Phi=sin(2pi(mu+nv))
        # \nabla\perp Phi = (-2 pi n cos(2pi(mu+nv)),2pi m cos(2pi(mu+nv)))
        matrix_dPhi[coeff+2, :, :, 0] = -2*np.pi * \
            n*np.cos(2*np.pi*(m*ugrid+n*vgrid))
        matrix_dPhi[coeff+2, :, :, 1] = 2*np.pi * \
            m*np.cos(2*np.pi*(m*ugrid+n*vgrid))
    return matrix_dPhi


def compute_j(boldpsi, matrixd_phi, Np):
    # the rotation matrix
    rot = np.array([[np.cos(2*np.pi/Np), -np.sin(2*np.pi/Np), 0],
                   [np.sin(2*np.pi/Np), np.cos(2*np.pi/Np), 0], [0, 0, 1]])
    (lc, lu, lv, _) = matrixd_phi.shape
    j = np.zeros((Np, lc, lu, lv, 3))
    j[0] = contract('ijkl,mkli->mklj', boldpsi, matrixd_phi)
    for i in range(1, Np):
        j[i] = contract('ij,klmj->klmi', rot, j[i-1])
    return j


def compute_dpsi_full(dpsi, Np):
    # the rotation matrix
    rot = np.array([[np.cos(2*np.pi/Np), -np.sin(2*np.pi/Np), 0],
                   [np.sin(2*np.pi/Np), np.cos(2*np.pi/Np), 0], [0, 0, 1]])
    (_, _, lu, lv) = dpsi.shape
    dpsi_full = np.zeros((Np, 2, 3, lu, lv))
    dpsi_full[0] = dpsi
    for i in range(1, Np):
        dpsi_full[i] = contract('ij,kjlm->kilm', rot, dpsi_full[i-1])
    return dpsi_full


def get_rot_tensor(Np):
    rot = np.array([[np.cos(2*np.pi/Np), -np.sin(2*np.pi/Np), 0],
                   [np.sin(2*np.pi/Np), np.cos(2*np.pi/Np), 0], [0, 0, 1]])
    rot_tensor = np.zeros((Np, 3, 3))
    for i in range(Np):
        rot_tensor[i] = np.linalg.matrix_power(rot, i)
    return rot_tensor


def compute_Qj(matrixd_phi, dpsi, dS):
    """take only the segment whitout rotation of j"""
    lu, lv = dS.shape
    Qj = contract('oija,adij,kdij,pijk,ij->op', matrixd_phi, dpsi,
                  dpsi, matrixd_phi, 1/dS, optimize=True)/(lu*lv)
    return Qj


def compute_dQjdtheta(matrixd_phi, dpsi, dS, dtheta, dSdtheta):
    """take only the segment whitout rotation of j"""
    lu, lv = dS.shape
    # derivation with respect to dS
    dQj = contract('oija,adij,kdij,pijk,zij->zop', matrixd_phi, dpsi,
                   dpsi, matrixd_phi, -dSdtheta/(dS*dS), optimize=True)/(lu*lv)
    # dtildetheta_symmetrized=dtildetheta+contract('zijed->zijde',dtildetheta)
    dQj += contract('oija,zijad,kdij,pijk,ij->zop', matrixd_phi, dtheta, dpsi, matrixd_phi, 1/dS, optimize=True)/(lu*lv) \
        + contract('oija,adij,zijkd,pijk,ij->zop', matrixd_phi, dpsi,
                   dtheta, matrixd_phi, 1/dS, optimize=True)/(lu*lv)
    return dQj


def compute_LS_old(T, j, dS, normalp):
    """T is the distance tensor Npxlu1xlv1xlu2xlv2 x 3
    and j the current tensor (lc+2) x Np x lu1 x lv1 x 3
    dS lu1 x lv1
    normalp 3 x lu2 x lv2"""
    lu1, lv1 = dS.shape
    D = 1/(np.linalg.norm(T, axis=-1)**3)
    # for cross product 
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    # the terrible formula...
    LS = contract('ijklmn,iojkp,qnp,jk,ijklm,qlm->olm', T,
                  j, eijk, dS, D, normalp, optimize=True)
    return (mu_0/(4*np.pi))*LS/(lu1*lv1)


def compute_LS(T, matrixd_phi, dpsi, rot_tensor, normalp):
    """T is the distance tensor Npxlu1xlv1xlu2xlv2 x 3
    and j the current tensor (lc+2) x Np x lu1 x lv1 x 3
    dS lu1 x lv1
    normalp 3 x lu2 x lv2"""
    _, lu1, lv1, _ = matrixd_phi.shape
    D = 1/(np.linalg.norm(T, axis=-1)**3)
    # for cross product
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    # the terrible formula...
    LS = contract('ijklmn,ojkw,ipz,wzjk,qnp,ijklm,qlm->olm', T,
                  matrixd_phi, rot_tensor,  dpsi, eijk, D, normalp, optimize=True)
    return (mu_0/(4*np.pi))*LS/(lu1*lv1)


def compute_dLS(T, matrixd_phi, dpsi, rot_tensor, normalp, theta, dtildetheta):
    """T is the distance tensor Npxlu1xlv1xlu2xlv2 x 3
    and j the current tensor (lc+2) x Np x lu1 x lv1 x 3
    dS lu1 x lv1
    normalp 3 x lu2 x lv2"""
    _, lu1, lv1, _ = matrixd_phi.shape
    D = 1/(np.linalg.norm(T, axis=-1)**3)
    DD = 1/(np.linalg.norm(T, axis=-1)**5)
    # for cross product
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    # the terrible formula...
    # first dT/dtheta
    dLS = -contract('inb,ajkb,ojkw,ipz,wzjk,qnp,ijklm,qlm->aolm', rot_tensor, theta,
                    matrixd_phi, rot_tensor, dpsi, eijk, D, normalp, optimize=True)/(lu1*lv1)
    # then the -3(x-y,h)(x-y)
    dLS += 3*contract('ibc,ajkc,ijklmb,ijklmn,ojkw,ipz,wzjk,qnp,ijklm,qlm->aolm', rot_tensor,
                      theta, T, T, matrixd_phi, rot_tensor, dpsi, eijk, DD, normalp, optimize=True)/(lu1*lv1)
    # finally the dpsi/dtheta part
    # dLS=contract('inb,ajkb,ojkw,ipz,wzjk,qnp,ijklm,qlm->aolm',rot_tensor,theta,matrixd_phi,rot_tensor,dpsi,eijk,D,normalp,optimize=True)
    dLS += contract('ijklmn,ojkw,ipz,ajkbz,wbjk,qnp,ijklm,qlm->aolm', T, matrixd_phi,
                    rot_tensor, dtildetheta, dpsi, eijk, D, normalp, optimize=True)/(lu1*lv1)
    return (mu_0/(4*np.pi))*dLS
