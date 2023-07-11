import logging

import numpy as np
import pytest

from stellacode.surface.fourier import *


@pytest.mark.skip("graphic verif skipping")
def test_normal_derivative():
    lu, lv = 128, 128
    logging.basicConfig(level="DEBUG")
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)

    import matplotlib.pyplot as plt

    grad_dS = np.gradient(S.ds, 1 / lu, 1 / lv)
    # plt.plot(grad_dS[1][33])
    # plt.plot(S.ds_v[33])
    # plt.show()

    grad_nx = np.gradient(S.n[0], 1 / lu, 1 / lv)
    grad_ny = np.gradient(S.n[1], 1 / lu, 1 / lv)
    grad_nz = np.gradient(S.n[2], 1 / lu, 1 / lv)
    # plt.plot(grad_nx[0][33])
    # plt.plot(S.n_u[0][33])
    plt.plot(grad_ny[1][33])
    plt.plot(S.n_v[1][33])
    plt.show()
