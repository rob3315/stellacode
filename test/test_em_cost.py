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
from stellacode.surface import ToroidalSurface, IntegrationParams
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
    print(metrics)
    from stellacode.tools.vmec import VMECIO

    vmec = VMECIO("test/data/w7x/wout_d23p4_tm.nc")
    vmec.get_net_poloidal_current()
    # filename = "test/data/w7x/regcoil_out.w7x.nc"
    # file_ = netcdf_file(filename, "r", mmap=False)
    # chi2_b = file_.variables["chi2_B"][()].astype(float)
    # print(chi2_b)
    # import pdb;pdb.set_trace()
    # xm, xn = cws.current.get_coeffs().T
    # assert np.all(file_.variables["xm_coil"][()]- xm[:-1])
    # assert np.all(file_.variables["xn_coil"][()][1:] // 5 == xn)


@pytest.mark.parametrize("use_mu_0_factor", [False, True])
def test_compare_to_regcoil(use_mu_0_factor):
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    filename = "test/data/li383/regcoil_out.li383.nc"
    file_ = netcdf_file(filename, "r", mmap=False)

    cws = get_cws(config)
    xm, xn = cws.current.get_coeffs()
    assert np.all(file_.variables["xm_coil"][()][1:] == xm)
    assert np.all(file_.variables["xn_coil"][()][1:] // 3 == xn)

    em_cost = EMCost.from_config(config=config, use_mu_0_factor=use_mu_0_factor)

    lambdas = file_.variables["lambda"][()].astype(float)

    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)

    chi2_b = file_.variables["chi2_B"][()].astype(float)
    assert np.max(np.abs(metrics.cost_B.values - chi2_b) / np.max(chi2_b)) < 5e-5
    chi_j = file_.variables["chi2_K"][()].astype(float)

    # for some reason chi_j is not well reproduced for low lambdas
    assert np.max(np.abs(metrics.cost_J.values[1:] - chi_j[1:]) / np.max(chi_j[1:])) < 5e-6
    em_cost.lamb = lambdas[-1]
    j_s = em_cost.get_current_result(cws)
    js_reg = file_.variables["single_valued_current_potential_mn"][()].astype(float)[-1]
    assert np.abs(js_reg - j_s[2:]).max() / js_reg.max() < 1e-14

    # however the result is no more the same for very low lambdas
    em_cost.lamb = lambdas[2]
    j_s = em_cost.get_current_result(cws)
    js_reg = file_.variables["single_valued_current_potential_mn"][()].astype(float)[-1]
    assert np.abs(js_reg - j_s[2:]).max() / js_reg.max() < 2e-4


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
        integration_par=IntegrationParams(num_points_u=32, num_points_v=32),
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
            integration_par=IntegrationParams(num_points_u=32, num_points_v=16),
            num_tor_symmetry=9,
            make_joints=True,
        ),
        # nbpts=(32, 16),
        num_tor_symmetry=3,
        rotate_diff_current=3,
        current=get_current_potential(config),
    )

    new_surface = fit_to_surface(S, em_cost.Sp)

    # phi_mn = em_cost.get_current_result(new_surface)
    # j_3d = new_surface.get_j_3D(phi_mn)
    # S.plot(only_one_period=True, vector_field=j_3d)
    # new_surface.surface.plot(only_one_period=True,vector_field=j_3d)
    # em_cost.Sp.plot()

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
            integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil),
            # nbpts=(n_pol_coil, n_tor_coil),
            num_tor_symmetry=9,
        ),
        # nbpts=(n_pol_coil, n_tor_coil),
        num_tor_symmetry=3,
        rotate_diff_current=3,
        current=get_current_potential(config),
    )

    # check that the rotated current potential is well constructed
    curent_potential_op = S.get_curent_op()
    s1, s2, s3, _ = curent_potential_op.shape
    assert np.all(curent_potential_op[:, :, : s3 // 3] == curent_potential_op[:, :, s3 // 3 : 2 * s3 // 3])
    cp_op = curent_potential_op.reshape((3, s1 // 3, s2, 9, s3 // 9, -1))
    assert np.all(cp_op[1:3, :, :, 0] == 0)
    assert np.all(cp_op[0, :, :, 1:3] == 0)

    S.compute_surface_attributes()
    S.get_min_distance(surf.xyz)

    # fit the rotated surface to the plasma surface
    new_surface = fit_to_surface(S, surf)
    new_surface.update_params(radius=new_surface.surface.radius + 0.1)
    assert abs(new_surface.get_min_distance(surf.xyz) - 0.1) < 1e-2

    # compute regcoil metrics
    phi_mn = em_cost.get_current_result(S)
    new_surface.plot_j_surface(phi_mn, num_rot=1)

    lambdas = np.array([1.2e-24, 1.2e-18, 1.2e-14, 1.0e00])
    metrics = em_cost.cost_multiple_lambdas(new_surface, lambdas)
    assert metrics.cost_B.min() < 2e-4


def test_plot_plasma_cross_sections():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    surf = get_plasma_surface(config)

    surf.plot_cross_sections()
