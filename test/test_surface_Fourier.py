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

    grad_dS = np.gradient(S.dS, 1 / lu, 1 / lv)
    # plt.plot(grad_dS[1][33])
    # plt.plot(S.dS_v[33])
    # plt.show()

    grad_nx = np.gradient(S.n[0], 1 / lu, 1 / lv)
    grad_ny = np.gradient(S.n[1], 1 / lu, 1 / lv)
    grad_nz = np.gradient(S.n[2], 1 / lu, 1 / lv)
    # plt.plot(grad_nx[0][33])
    # plt.plot(S.n_u[0][33])
    plt.plot(grad_ny[1][33])
    plt.plot(S.n_v[1][33])
    plt.show()


@pytest.mark.skip("get_theta_pertubation")
def test_perimeter_derivative():
    lu, lv = 128, 128
    eps = 1e-8
    logging.basicConfig(level="DEBUG")
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)
    params = S.params
    ls = len(params[0])  # total number of hamonics
    # S=Surface_Fourier(params,(lu,lv),3)
    theta_pertubation = S.get_theta_pertubation()
    dtildetheta = theta_pertubation["dtildetheta"]
    perim = np.einsum("ij->", S.dS) / (lu * lv)
    perturb = 2 * np.random.random(2 * ls) - 1
    new_param = FourierSurface.change_param(params, eps * perturb)
    new_S = FourierSurface(new_param, (lu, lv), 3)
    new_perim = np.einsum("ij->", new_S.dS) / (lu * lv)

    dperim_num = (new_perim - perim) / eps
    dperim = np.einsum("cijll,c,ij", dtildetheta, perturb, S.dS) / (lu * lv)
    np.testing.assert_almost_equal(dperim, dperim_num, decimal=3)
    print(dperim_num, dperim)


@pytest.mark.skip("get_theta_pertubation")
def test_change_of_variable_scalar():
    lu, lv = 128, 128
    eps = 1e-8
    logging.basicConfig(level="DEBUG")
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)
    params = S.params
    ls = len(params[0])  # total number of hamonics
    # S=Surface_Fourier(params,(lu,lv),3)
    theta_pertubation = S.get_theta_pertubation()
    theta = theta_pertubation["theta"]
    dtildetheta = theta_pertubation["dtildetheta"]
    f = np.random.random((lu, lv))
    int_f = np.einsum("ij,ij->", f, S.dS) / (lu * lv)
    perturb = 2 * np.random.random(2 * ls) - 1
    new_param = FourierSurface.change_param(params, eps * perturb)
    new_S = FourierSurface(new_param, (lu, lv), 3)
    new_int_f = np.einsum("ij,ij->", new_S.dS, f) / (lu * lv)

    dint_num = (new_int_f - int_f) / eps
    dint = np.einsum("cijll,c,ij,ij", dtildetheta, perturb, S.dS, f) / (lu * lv)
    np.testing.assert_almost_equal(dint, dint_num, decimal=3)
    print(dint_num, dint)


@pytest.mark.skip("get_theta_pertubation")
def test_dSdtheta():
    lu, lv = 128, 128
    eps = 1e-8
    logging.basicConfig(level="DEBUG")
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)
    params = S.params
    ls = len(params[0])  # total number of hamonics
    # S=Surface_Fourier(params,(lu,lv),3)

    theta_pertubation = S.get_theta_pertubation()
    dSdtheta = theta_pertubation["dSdtheta"]
    perturb = 2 * np.random.random(2 * ls) - 1
    new_param = FourierSurface.change_param(params, eps * perturb)
    new_S = FourierSurface(new_param, (lu, lv), 3)

    ddS_num = (new_S.dS - S.dS) / eps
    ddS = np.einsum("cij,c", dSdtheta, perturb)
    np.testing.assert_almost_equal(ddS_num, ddS, decimal=2)


def test_curvature_derivative():
    lu, lv = 128, 128
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)
    S = FourierSurface.from_file("data/li383/cws.txt", 3, lu, lv)
    # S=Surface_Fourier(params,(lu,lv),3)
