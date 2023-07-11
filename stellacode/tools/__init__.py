import numpy as onp

from stellacode import np

# the completely antisymetric tensor
eijk = onp.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
eijk = np.asarray(eijk)


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
