import configparser

import jax

jax.config.update("jax_enable_x64", True)
import pytest
from scipy.io import netcdf_file

from stellacode import np
from stellacode.costs.em_cost import EMCost
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.imports import (
    get_current_potential,
    get_cws,
    get_plasma_surface,
)
from stellacode.surface.rotated_surface import RotatedSurface
from stellacode.surface.tore import ToroidalSurface
from stellacode.surface.utils import fit_to_surface


def test_no_dimension_error():
    ###just check that all operations respects dimensions
    path_config_file = "config_file/config_test_dim.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    EMCost.from_config_file(path_config_file)


def test_compare_to_matlab_regcoil():
    path_config_file = "test/data/w7x/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    cws = get_cws(config)
    em_cost = EMCost.from_config(config=config, use_mu_0_factor=True)

    metrics = em_cost.cost(cws)


@pytest.mark.parametrize("use_mu_0_factor", [False, True])
def test_compare_to_regcoil(use_mu_0_factor):
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    filename = "test/data/li383/regcoil_out.li383.nc"
    file_ = netcdf_file(filename, "r", mmap=False)

    cws = get_cws(config)
    em_cost = EMCost.from_config(config=config, use_mu_0_factor=use_mu_0_factor)

    lambdas = file_.variables["lambda"][()].astype(float)
    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)

    chi2_b = file_.variables["chi2_B"][()].astype(float)
    assert np.max(np.abs(metrics.cost_B.values - chi2_b) / np.max(chi2_b)) < 5e-5
    chi_j = file_.variables["chi2_K"][()].astype(float)
    assert np.max(np.abs(metrics.cost_J.values[1:] - chi_j[1:]) / np.max(chi_j[1:])) < 5e-6

    # chi2_b = file_.variables["chi2_B"][()][1:].astype(float)
    # assert np.max(np.abs(metrics.cost_B.values - chi2_b) / chi2_b) < 5e-5
    # chi_j = file_.variables["chi2_K"][()][1:].astype(float)
    # assert np.max(np.abs(metrics.cost_J.values - chi_j) / chi_j) < 5e-6


def test_regcoil_with_axisymmetric():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    em_cost = EMCost.from_config(config=config, use_mu_0_factor=False)
    major_radius = em_cost.Sp.get_major_radius()
    minor_radius = em_cost.Sp.get_minor_radius()

    S = ToroidalSurface(
        num_tor_symmetry=3,
        major_radius=major_radius,
        minor_radius=minor_radius + 0.2,
        params={},
        nbpts=(32, 32),
    )
    S.get_min_distance(em_cost.Sp.xyz)

    assert abs(S.get_min_distance(em_cost.Sp.xyz) - 0.2) < 2e-3


def test_pwc_fit():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    em_cost = EMCost.from_config(config=config, use_mu_0_factor=False)

    fourier_coeffs = np.zeros((5, 2))

    S = RotatedSurface(
        surface=CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            nbpts=(32, 16),
            num_tor_symmetry=9,
            make_joints=True,
        ),
        nbpts=(32, 16),
        num_tor_symmetry=3,
        rotate_diff_current=3,
        current=get_current_potential(config),
    )

    # S.plot(only_one_period=True)
    # S.surface.plot(only_one_period=True)

    new_surface = fit_to_surface(S, em_cost.Sp)

    assert new_surface.get_min_distance(em_cost.Sp.xyz) < 3e-2


def test_regcoil_with_pwc():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    em_cost = EMCost.from_config(config=config, use_mu_0_factor=False)

    surf = get_plasma_surface(config)

    fourier_coeffs = np.zeros((5, 2))
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])

    S = RotatedSurface(
        surface=CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            nbpts=(n_pol_coil, n_tor_coil),
            num_tor_symmetry=9,
        ),
        nbpts=(n_pol_coil, n_tor_coil),
        num_tor_symmetry=3,
        rotate_diff_current=3,
        current=get_current_potential(config),
    )

    # check that the rotated current potential is well constructed
    curent_potential_op = S.get_curent_op()
    s1, s2, s3, _ = curent_potential_op.shape
    assert s2 * 9 == s3

    assert np.all(curent_potential_op[:, :, : s3 // 3] == curent_potential_op[:, :, s3 // 3 : 2 * s3 // 3])
    cp_op = curent_potential_op.reshape((3, s1 // 3, s2, 9, s3 // 9, -1))
    assert np.all(cp_op[1:3, :, :, 0] == 0)
    assert np.all(cp_op[0, :, :, 1:3] == 0)

    S.compute_surface_attributes()
    S.get_min_distance(surf.xyz)

    # fit the rotated surface to the plasma surface
    new_surface = fit_to_surface(S, surf)
    new_surface.surface = new_surface.surface.copy(
        update=dict(
            radius=new_surface.surface.radius + 0.1,
        )
    )
    new_surface.compute_surface_attributes()
    assert (new_surface.get_min_distance(surf.xyz) - 0.1) < 1e-2

    # compute regcoil metrics

    # need to update matrix_dphi with new surface params
    em_cost.matrixd_phi = new_surface.get_curent_op()

    lambdas = np.array([1.2e-24, 1.2e-18, 1.2e-14, 1.0e00])
    metrics = em_cost.cost_multiple_lambdas(new_surface, lambdas)
    assert metrics.cost_B.min() < 2e-4


def test_plot_plasma_cross_sections():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    surf = get_plasma_surface(config)

    surf.plot_cross_sections()
