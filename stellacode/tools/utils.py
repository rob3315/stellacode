import jax.numpy as np
from scipy.integrate import quad
from scipy.spatial.distance import cdist

from stellacode import np


def get_min_dist(S1, S2):
    return cdist(np.reshape(S1, (-1, 3)), np.reshape(S2, (-1, 3))).min()

    # slower but differentiable
    # return np.linalg.norm(S1.P[...,None,None,:]-S2.P[None,None,...], axis=-1).min()


def fourier_coefficients(li, lf, n, f):
    # from: https://www.bragitoff.com/2021/05/fourier-series-coefficients-and-visualization-python-program/
    l = (lf - li) / 2
    # Constant term
    a0 = 1 / l * quad(lambda x: f(x), li, lf)[0]
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
