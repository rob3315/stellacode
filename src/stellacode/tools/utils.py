import jax.numpy as np
import numpy as onp
from scipy.integrate import quad
from scipy.spatial.distance import cdist

from stellacode import np


# the completely antisymetric tensor
def get_eijk():
    eijk = onp.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    eijk = np.asarray(eijk)
    return eijk


eijk = get_eijk()


def get_min_dist(S1, S2):
    return cdist(np.reshape(S1, (-1, 3)), np.reshape(S2, (-1, 3))).min()

    # slower but differentiable
    # return np.linalg.norm(S1.P[...,None,None,:]-S2.P[None,None,...], axis=-1).min()


def fourier_coefficients(li, lf, n, fun):
    """Compute the n first fourier coefficients of the function fun from li to lf"""
    # from: https://www.bragitoff.com/2021/05/fourier-series-coefficients-and-visualization-python-program/
    l = (lf - li) / 2
    # Constant term
    a0 = 1 / l * quad(lambda x: fun(x), li, lf)[0]
    # Cosine coefficents
    A = np.zeros((n))
    # Sine coefficents
    B = np.zeros((n))

    coefs = []
    for i in range(1, n + 1):
        omega = i * np.pi / l
        A = quad(lambda x: f(x) * np.cos(omega * x), li, lf)[0] / l
        B = quad(lambda x: f(x) * np.sin(omega * x), li, lf)[0] / l
        coefs.append(np.array([A, B]))

    return a0 / 2.0, np.stack(coefs, axis=0)


def cfourier1D(nzeta, fmnc, xn):
    cfunct = []
    for i in range(nzeta):
        ci = 0
        zeta = i*2*np.pi/nzeta
        for x, mode in zip(xn, fmnc):
            angle = - x*zeta
            ci += mode*np.cos(angle)
        cfunct.append(ci)
    return np.array(cfunct)


def sfourier1D(nzeta, fmns, xn):
    sfunct = []
    for i in range(nzeta):
        si = 0
        zeta = i*2*np.pi/nzeta
        for x, mode in zip(xn, fmns):
            angle = - x*zeta
            si += mode*np.sin(angle)
        sfunct.append(si)
    return np.array(sfunct)
