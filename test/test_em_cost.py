import configparser

import jax

jax.config.update("jax_enable_x64", True)
import pytest
from scipy.io import netcdf_file

from stellacode import np
from stellacode.costs.em_cost import EMCost, get_b_field_err
from stellacode.definitions import w7x_plasma, ncsx_plasma
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface import CylindricalSurface, FourierSurface
from stellacode.surface.imports import (
    get_current_potential,
    get_cws,
    get_net_current,
    get_plasma_surface,
    get_cws_from_plasma_config,
)
from stellacode.surface.rotated_surface import RotatedSurface
from stellacode.surface.utils import fit_to_surface
from stellacode.tools.vmec import VMECIO


def test_no_dimension_error():
    ###just check that all operations respects dimensions
    path_config_file = "config_file/config_test_dim.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    EMCost.from_config_file(path_config_file)


def test_reproduce_regcoil_axisym():
    major_radius = 5.5
    minor_radius = 1.404687741189692  # 0.9364584941264614*(1+0.5)

    current = Current(num_pol=8, num_tor=8, net_currents=get_net_current(w7x_plasma.path_plasma))

    cws = RotatedSurface(
        surface=ToroidalSurface(
            num_tor_symmetry=5,
            major_radius=major_radius,
            minor_radius=minor_radius,
            params={},
            integration_par=current.get_integration_params(),
        ),
        num_tor_symmetry=5,
        rotate_diff_current=1,
        current=current,
    )

    em_cost = EMCost.from_plasma_config(
        plasma_config=w7x_plasma, integration_par=IntegrationParams(num_points_u=32, num_points_v=32)
    )

    filename = "test/data/w7x/regcoil_out.w7x_axis.nc"
    file_ = netcdf_file(filename, "r", mmap=False)
    lambdas = file_.variables["lambda"][()].astype(float)
    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)[0]

    # the agreement is not perfect for very low lambdas
    chi_b = file_.variables["chi2_B"][()].astype(float)
    assert np.all((np.abs(chi_b - metrics.cost_B.values)) / chi_b < 5e-3)
    chi_j = file_.variables["chi2_K"][()].astype(float)
    assert np.all((np.abs(chi_j - metrics.cost_J.values)) / chi_j < 5e-3)

    vmec = VMECIO.from_grid("test/data/w7x/wout_d23p4_tm.nc")
    assert np.abs(file_.variables["curpol"][()].astype(float) - vmec.curpol) < 1e-19
    pol_cur = file_.variables["net_poloidal_current_Amperes"][()].astype(float)
    assert np.abs(pol_cur - vmec.net_poloidal_current) / pol_cur < 1e-9


@pytest.mark.parametrize("use_mu_0_factor", [False, True])
def test_compare_to_regcoil(use_mu_0_factor):
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    filename = "test/data/li383/regcoil_out.li383.nc"
    file_ = netcdf_file(filename, "r", mmap=False)

    cws = get_cws(config)
    xm, xn = cws.current._get_coeffs()
    assert np.all(file_.variables["xm_coil"][()][1:] == xm)
    assert np.all(file_.variables["xn_coil"][()][1:] // 3 == xn)

    em_cost = EMCost.from_config(config=config, use_mu_0_factor=use_mu_0_factor)

    lambdas = file_.variables["lambda"][()].astype(float)

    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)[0]

    # for some reason chi_j is not well reproduced for low lambdas
    chi2_b = file_.variables["chi2_B"][()].astype(float)
    assert np.all((np.abs(metrics.cost_B.values - chi2_b) / chi2_b)[1:] < 5e-5)
    chi_j = file_.variables["chi2_K"][()].astype(float)
    assert np.all(np.abs(metrics.cost_J.values[1:] - chi_j[1:]) / chi_j[1:] < 5e-6)

    em_cost.lamb = lambdas[-1]
    solver = em_cost.get_regcoil_solver(cws)[0]
    phi_mn = solver.solve_lambda(lambdas[-1])
    js_reg = -file_.variables["single_valued_current_potential_mn"][()].astype(float)[-1]
    assert np.abs(js_reg - phi_mn[2:]).max() / js_reg.max() < 1e-14

    # however the result is no more the same for very low lambdas
    phi_mn = solver.solve_lambda(lambdas[2])
    js_reg = -file_.variables["single_valued_current_potential_mn"][()].astype(float)[-1]
    assert np.abs(js_reg - phi_mn[2:]).max() / js_reg.max() < 2.7e-4
    j_3d = cws.get_j_3D(phi_mn)

    phi_mn = solver.solve_lambda(1e-30)
    bs = em_cost.get_bs_operator(cws, normal_b_field=False)

    # Plot regcoil vs stellacode
    # import pandas as pd
    # import matplotlib.pyplot as plt;import seaborn as sns;import matplotlib;matplotlib.use('TkAgg')
    # from coil_optim.plots import format_axis
    # df = pd.concat({"regcoil": pd.Series(chi2_b, index=chi_j),"stellacode": pd.Series(metrics.cost_B.values, index=chi_j),}, axis=1)
    # ax=df.iloc[11:].plot(marker='.', linestyle='dashed')
    # ax.set_xlabel( r"$\chi^2_K$")
    # ax.set_ylabel(r"$\chi^2_B$")
    # format_axis(ax)
    # plt.show()
    # # for comparing the errors
    # b_err = np.linalg.norm(b_field_gt - np.transpose(b_field, (1, 2, 0)), axis=-1)
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # f, axes=plt.subplots(3,1)
    # sns.heatmap(b_err, ax=axes[0])
    # sns.heatmap(np.linalg.norm(b_field_gt, axis=-1), ax=axes[1])
    # sns.heatmap(np.linalg.norm(np.transpose(b_field, (1, 2, 0)), axis=-1), ax=axes[2])
    # plt.show()


@pytest.mark.parametrize("plasma_config", [w7x_plasma, ncsx_plasma])
@pytest.mark.parametrize("surface_label", [1, -1])
def test_b_field_err(plasma_config, surface_label):
    current_n_coeff = 4
    n_points = current_n_coeff * 6
    em_cost = EMCost.from_plasma_config(
        plasma_config=plasma_config,
        integration_par=IntegrationParams(num_points_u=n_points, num_points_v=n_points),
        lamb=1e-30,
        surface_label=surface_label,
        fit_b_3d=False
    )
    cws = get_cws_from_plasma_config(plasma_config, n_harmonics_current=current_n_coeff)

    b_field = em_cost.get_b_field(cws)
    b_field_gt = em_cost.Sp.get_gt_b_field(surface_labels=surface_label)[:, : em_cost.Sp.integration_par.num_points_v]

    err_b = em_cost.Sp.integrate((np.linalg.norm(b_field - b_field_gt, axis=-1))) / em_cost.Sp.integrate(
        (np.linalg.norm(b_field_gt, axis=-1))
    )

    if surface_label == -1:
        assert err_b < 0.09
    else:
        assert err_b < 0.29

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # _, axes = plt.subplots(3)

    # sns.heatmap(np.linalg.norm(b_field_gt, axis=-1), cmap="seismic", ax=axes[0])
    # sns.heatmap(np.linalg.norm(b_field, axis=-1), cmap="seismic", ax=axes[1])
    # sns.heatmap(np.linalg.norm(b_field_gt - b_field, axis=-1), cmap="seismic", ax=axes[2], center=0)
    # plt.show()


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

    # phi_mn = em_cost.get_current_weights(new_surface)
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
    curent_potential_op = S.current_op[2:]
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
    phi_mn = em_cost.get_current_weights(S)
    new_surface.plot_j_surface(phi_mn, num_rot=1)

    # check that current basis has no net currents except in the first two functions
    curr_op = S.current._get_matrix_from_grid(S.grids)
    assert np.abs(curr_op[..., 0].sum(2))[2:].max() < 1e-11
    assert np.abs(curr_op[..., 1].sum(1))[2:].max() < 1e-11


def test_regcoil_with_pwc_no_current_at_bc():
    current_n_coeff = 8
    n_points = current_n_coeff * 4
    em_cost = EMCost.from_plasma_config(
        plasma_config=w7x_plasma,
        integration_par=IntegrationParams(num_points_u=n_points, num_points_v=n_points),
        lamb=1e-20,
    )

    current = CurrentZeroTorBC(
        num_pol=current_n_coeff,
        num_tor=current_n_coeff,
        sin_basis=True,
        cos_basis=True,
        net_currents=get_net_current(w7x_plasma.path_plasma),
    )

    fourier_coeffs = np.zeros((0, 2))

    coil_surf = RotatedSurface(
        surface=CylindricalSurface(
            fourier_coeffs=fourier_coeffs,
            integration_par=IntegrationParams(num_points_u=n_points, num_points_v=n_points),
            num_tor_symmetry=9,
        ),
        num_tor_symmetry=3,
        rotate_diff_current=3,
        current=current,
    )

    # fit the rotated surface to the plasma surface
    new_surface = fit_to_surface(coil_surf, em_cost.Sp)
    new_surface.update_params(radius=new_surface.surface.radius + 0.3)

    phi_mn = em_cost.get_current_weights(new_surface)
    j_s = new_surface.get_j_surface(phi_mn)

    # exactly at the boundary, the current is zero
    assert np.max(np.abs(j_s[:, 0, 1]) / np.max(np.abs(j_s))) < 0.1

    j_3d = new_surface.get_j_3D(phi_mn)
    # new_surface.plot(only_one_period=True, vector_field=j_3d)
    # new_surface.surface.plot(only_one_period=False, vector_field=j_3d, detach_parts=True)


def test_plot_plasma_cross_sections():
    surf = FourierSurface.from_file(
        ncsx_plasma.path_plasma,
        integration_par=IntegrationParams(num_points_u=32, num_points_v=32),
        n_fp=3,
    )
    surf.plot_cross_sections(num=5)
    # import matplotlib.pyplot as plt
    # plt.show()
