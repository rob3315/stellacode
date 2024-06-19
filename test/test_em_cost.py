import matplotlib
from stellacode.tools.vmec import VMECIO
from stellacode.surface.utils import fit_to_surface
from stellacode.surface.imports import (
    get_current_potential,
    get_cws,
    get_cws_from_plasma_config,
    get_net_current,
    get_plasma_surface,
)
from stellacode.surface.factory_tools import Sequential, rotate_coil
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    CylindricalSurface,
    FourierSurfaceFactory,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.definitions import ncsx_plasma, w7x_plasma
from stellacode.costs.em_cost import EMCost, get_b_field_err
from stellacode import np, PROJECT_PATH
from scipy.io import netcdf_file
import pytest
from os.path import dirname, join, realpath
import configparser

import jax
matplotlib.use('Agg')
jax.config.update("jax_enable_x64", True)

TEST_FOLDER_PATH = f"{(dirname(realpath(__file__)))}"
TEST_CONFIG_PATH = join(
    PROJECT_PATH, "config_file", "config_test_dim.ini")


def test_no_dimension_error():
    """
    Test that all operations respect dimensions.

    This test checks that all operations in the EMCost class respect dimensions by
    creating an instance of the EMCost class from a configuration file that contains
    the dimensions of the plasma surface. If there are no dimension errors, the test
    passes.
    """

    # Read the configuration file that contains the dimensions of the plasma surface
    config = configparser.ConfigParser()
    config.read(TEST_CONFIG_PATH)

    # Create an instance of the EMCost class from the configuration file
    # If there are no dimension errors, the test passes
    EMCost.from_config_file(TEST_CONFIG_PATH)


def test_reproduce_regcoil_axisym():
    """
    Test the agreement of the EM cost calculation with regcoil for an axisymmetric case.

    This test reproduces the results of the axisymmetric case in regcoil. It uses the
    parameters from the regcoil output file to calculate the EM cost using the EMCost class.
    The results are compared to the regcoil output file and the agreement is checked
    for various values of lambda.

    """

    # Define the plasma geometry
    major_radius = 5.5
    minor_radius = 1.404687741189692  # 0.9364584941264614*(1+0.5)

    # Create a current object with the net currents from regcoil output file
    current = Current(num_pol=8, num_tor=8,
                      net_currents=get_net_current(w7x_plasma.path_plasma))

    # Create a toroidal surface factory object with the plasma geometry
    surface = ToroidalSurface(
        nfp=5,
        major_radius=major_radius,
        minor_radius=minor_radius,
        integration_par=current.get_integration_params(),
    )

    # Create a coil surface factory object with the current and the surface factory
    cws = Sequential(
        surface_factories=[
            surface,
            rotate_coil(
                current=current,
                nfp=5,
            ),
        ]
    )()

    # Create an instance of the EMCost class for the axisymmetric plasma
    em_cost = EMCost.from_plasma_config(
        plasma_config=w7x_plasma,
        integration_par=IntegrationParams(
            num_points_u=32, num_points_v=32),
    )

    # Read the regcoil output file
    filename = join(TEST_FOLDER_PATH,
                    "data", "w7x", "regcoil_out.w7x_axis.nc")
    file_ = netcdf_file(filename, "r", mmap=False)

    # Calculate the EM cost for the coil surface factory at the lambda values in the regcoil output file
    # There is a different definition of lambda between regcoil and stellacode:
    # lambda_regcoil= lambda_stellacode*num_field_period
    lambdas = file_.variables["lambda"][()].astype(float) / 5
    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)[0]

    # Compare the calculated EM cost with the regcoil output file
    chi_b = file_.variables["chi2_B"][()].astype(float)
    assert np.all((np.abs(chi_b - metrics.cost_B.values)) / chi_b < 5e-3)
    chi_j = file_.variables["chi2_K"][()].astype(float)
    assert np.all((np.abs(chi_j - metrics.cost_J.values)) / chi_j < 5e-3)

    # Compare the net poloidal current with the regcoil output file
    vmec = VMECIO.from_grid(
        join(TEST_FOLDER_PATH, "data", "w7x", "wout_d23p4_tm.nc"))
    assert np.abs(file_.variables["curpol"]
                  [()].astype(float) - vmec.curpol) < 1e-19
    pol_cur = file_.variables["net_poloidal_current_Amperes"][()].astype(float)
    assert np.abs(pol_cur - vmec.net_poloidal_current) / pol_cur < 1e-9


@pytest.mark.parametrize("use_mu_0_factor", [False, True])
def test_compare_to_regcoil(use_mu_0_factor):
    """
    Compare the calculated EM cost with the regcoil output file for a coil surface factory.

    Parameters:
    use_mu_0_factor (bool): If True, use the mu_0 factor in the calculation of the EM cost.
    """

    # Load the configuration file
    path_config_file = join(
        TEST_FOLDER_PATH, "data", "li383", "config.ini")
    config = configparser.ConfigParser()
    config.read(path_config_file)

    # Read the regcoil output file
    filename = join(TEST_FOLDER_PATH, "data", "li383", "regcoil_out.li383.nc")
    file_ = netcdf_file(filename, "r", mmap=False)

    # Create an instance of the CoilSurfaceFactory class
    cws_factory = get_cws(config)
    cws = cws_factory()

    # Compare the coil surface factory coefficients with the regcoil output file
    xm, xn = cws_factory.surface_factories[1].current._get_coeffs()
    assert np.all(file_.variables["xm_coil"][()][1:] == xm)
    assert np.all(file_.variables["xn_coil"][()][1:] // 3 == xn)

    # Create an instance of the EMCost class for the coil surface factory
    em_cost = EMCost.from_config(
        config=config, use_mu_0_factor=use_mu_0_factor)

    # Calculate the EM cost for the coil surface factory at the lambda values in the regcoil output file
    lambdas = file_.variables["lambda"][()].astype(float) / 3
    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)[0]

    # Compare the calculated EM cost with the regcoil output file
    # for some reason chi_j is not well reproduced for low lambdas
    chi2_b = file_.variables["chi2_B"][()].astype(float)
    assert np.all((np.abs(metrics.cost_B.values - chi2_b) / chi2_b)[1:] < 5e-5)
    chi_j = file_.variables["chi2_K"][()].astype(float)
    assert np.all(
        np.abs(metrics.cost_J.values[1:] - chi_j[1:]) / chi_j[1:] < 5e-6)

    # Compare the net poloidal current with the regcoil output file
    em_cost.lamb = lambdas[-1]
    solver = em_cost.get_regcoil_solver(cws)
    phi_mn = solver.solve_lambda(lambdas[-1])
    js_reg = - \
        file_.variables["single_valued_current_potential_mn"][()
                                                              ].astype(float)[-1]
    assert np.abs(js_reg - phi_mn[2:]).max() / js_reg.max() < 1e-14

    # Compare the result for very low lambdas
    phi_mn = solver.solve_lambda(lambdas[2])
    js_reg = - \
        file_.variables["single_valued_current_potential_mn"][()
                                                              ].astype(float)[-1]
    assert np.abs(js_reg - phi_mn[2:]).max() / js_reg.max() < 2.7e-4
    j_3d = cws.get_j_3d(phi_mn)

    # Compare the result for very low lambdas
    phi_mn = solver.solve_lambda(1e-30)
    bs = em_cost.get_bs_operator(cws, normal_b_field=False)


@pytest.mark.parametrize("plasma_config", [w7x_plasma, ncsx_plasma])
@pytest.mark.parametrize("surface_label", [1, -1])
def test_b_field_err(plasma_config, surface_label):
    """
    Test if the calculated magnetic field from the EM cost class matches the ground truth.

    Parameters
    ----------
    plasma_config : PlasmaConfig
        The plasma configuration.
    surface_label : int
        The surface label.
    """
    # Define the number of Fourier coefficients for the current
    current_n_coeff = 4

    # Define the number of points for integration
    n_points = current_n_coeff * 6

    # Create the EM cost object
    em_cost = EMCost.from_plasma_config(
        plasma_config=plasma_config,
        integration_par=IntegrationParams(
            num_points_u=n_points, num_points_v=n_points),
        lamb=1e-30,
        surface_label=surface_label,
        fit_b_3d=False,
    )

    # Create the coil object from the plasma configuration and number of Fourier coefficients
    cws = get_cws_from_plasma_config(
        plasma_config, n_harmonics_current=current_n_coeff)

    # Calculate the magnetic field using the EM cost object
    b_field = em_cost.get_b_field(cws())

    # Get the ground truth magnetic field
    b_field_gt = em_cost.Sp.get_gt_b_field(surface_labels=surface_label)[
        :, : em_cost.Sp.integration_par.num_points_v]

    # Calculate the error between the calculated and ground truth magnetic field
    err_b = em_cost.Sp.integrate((np.linalg.norm(b_field - b_field_gt, axis=-1))) / em_cost.Sp.integrate(
        (np.linalg.norm(b_field_gt, axis=-1))
    )

    # Check if the error is within the acceptable range
    if surface_label == -1:
        assert err_b < 0.09
    else:
        assert err_b < 0.29

    # Uncomment the following code to plot the comparison between the calculated and ground truth magnetic field
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # _, axes = plt.subplots(3)

    # sns.heatmap(np.linalg.norm(b_field_gt, axis=-1), cmap="seismic", ax=axes[0])
    # sns.heatmap(np.linalg.norm(b_field, axis=-1), cmap="seismic", ax=axes[1])
    # sns.heatmap(np.linalg.norm(b_field_gt - b_field, axis=-1), cmap="seismic", ax=axes[2], center=0)
    # plt.show()


def test_regcoil_with_axisymmetric():
    """
    Test if the computed minimum distance between a toroidal surface and a plasma surface
    is approximately equal to the expected distance.
    """
    # Construct the path to the configuration file
    path_config_file = join(TEST_FOLDER_PATH, "data", "li383", "config.ini")

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(path_config_file)

    # Create an instance of the EMCost class from the configuration file
    em_cost = EMCost.from_config(config=config, use_mu_0_factor=False)

    # Get the major and minor radii of the plasma surface
    major_radius = em_cost.Sp.get_major_radius()
    minor_radius = em_cost.Sp.get_minor_radius()

    # Create a toroidal surface with the same number of field periods (nfp) as the plasma surface
    # and a minor radius increased by 0.2
    S = ToroidalSurface(
        nfp=3,
        major_radius=major_radius,
        minor_radius=minor_radius + 0.2,
        integration_par=IntegrationParams(num_points_u=32, num_points_v=32),
    )()

    # Compute the minimum distance between the toroidal surface and the plasma surface
    S.get_min_distance(em_cost.Sp.xyz)

    # Check if the computed minimum distance is approximately equal to the expected distance
    assert abs(S.get_min_distance(em_cost.Sp.xyz) - 0.2) < 2e-3


def test_pwc_fit():
    """
    Test if the computed minimum distance between a cylindrical surface and a plasma surface
    is less than 3e-2.
    """
    # Construct the path to the configuration file
    path_config_file = join(TEST_FOLDER_PATH, "data", "li383", "config.ini")

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(path_config_file)

    # Create an instance of the EMCost class from the configuration file
    em_cost = EMCost.from_config(config=config, use_mu_0_factor=False)

    # Create a cylindrical surface with Fourier coefficients of zero
    fourier_coeffs = np.zeros((5, 2))
    surface = CylindricalSurface(
        fourier_coeffs=fourier_coeffs,
        integration_par=IntegrationParams(num_points_u=32, num_points_v=16),
        nfp=9,
        make_joints=True,
    )

    # Create a Sequential surface factory with the cylindrical surface and a rotated coil
    S = Sequential(
        surface_factories=[
            surface,
            rotate_coil(current=get_current_potential(
                config), nfp=3, num_surf_per_period=3),
        ]
    )

    # Fit the Sequential surface factory to the plasma surface
    new_surface = fit_to_surface(S, em_cost.Sp)

    # Check if the computed minimum distance is less than 3e-2
    assert new_surface().get_min_distance(em_cost.Sp.xyz) < 3e-2


def test_regcoil_with_pwc():
    """
    Tests if rotated current potential is well constructed and if the fitted surface
    has the correct minimum distance to the plasma surface. Also computes regcoil metrics
    and plots them.
    """
    # Construct the path to the configuration file
    path_config_file = join(TEST_FOLDER_PATH, "data", "li383", "config.ini")

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(path_config_file)

    # Create an instance of the EMCost class from the configuration file
    em_cost = EMCost.from_config(config=config, use_mu_0_factor=False)

    # Create a plasma surface
    surf = get_plasma_surface(config)()

    # Create a cylindrical surface with Fourier coefficients of zero
    fourier_coeffs = np.zeros((5, 2))
    n_pol_coil = int(config["geometry"]["ntheta_coil"])
    n_tor_coil = int(config["geometry"]["nzeta_coil"])
    surface = CylindricalSurface(
        fourier_coeffs=fourier_coeffs,
        integration_par=IntegrationParams(
            num_points_u=n_pol_coil, num_points_v=n_tor_coil),
        nfp=9,
    )

    # Create a Sequential surface factory with the cylindrical surface and a rotated coil
    factory = Sequential(
        surface_factories=[
            surface,
            rotate_coil(current=get_current_potential(
                config), nfp=3, num_surf_per_period=3),
        ]
    )

    # Create the surface
    S = factory()

    # Check if the rotated current potential is well constructed
    curent_potential_op = S.current_op[2:]
    s1, s2, s3, _ = curent_potential_op.shape
    assert np.all(curent_potential_op[:, :, : s3 // 3]
                  == curent_potential_op[:, :, s3 // 3: 2 * s3 // 3])
    cp_op = curent_potential_op.reshape((3, s1 // 3, s2, 9, s3 // 9, -1))
    assert np.all(cp_op[1:3, :, :, 0] == 0)
    assert np.all(cp_op[0, :, :, 1:3] == 0)

    # Compute minimum distance to the plasma surface
    S.get_min_distance(surf.xyz)

    # Fit the rotated surface to the plasma surface
    new_surface = fit_to_surface(factory, surf)
    new_surface.surface_factories[0].update_params(
        radius=new_surface.surface_factories[0].radius + 0.1)
    assert abs(new_surface().get_min_distance(surf.xyz) - 0.1) < 1e-2

    # Compute regcoil metrics
    phi_mn = em_cost.get_current_weights(S)

    # TODO: add later
    # new_surface().plot_j_surface(phi_mn)


def test_current_conservation():
    """
    Test that the current basis has no net currents except in the first two functions.
    """
    plasma_config = w7x_plasma
    import numpy as onp

    # create a surface factory with the plasma config and 4 harmonics for the current
    factory = get_cws_from_plasma_config(plasma_config, n_harmonics_current=4)

    # set the current coefficient to random values and compute the surface
    current = factory.surface_factories[1].surface_factories[0].current
    current.set_phi_mn(onp.random.random(current.phi_mn.shape) * 1e5)
    cws = factory()

    # check that the current operator summed over each direction is zero except in the first two functions
    curr_op = cws.current_op
    assert np.abs(curr_op[..., 0].sum(2))[2:].max() < 1e-11
    assert np.abs(curr_op[..., 1].sum(1))[2:].max() < 1e-11

    # read the vmec output grid
    vmec = VMECIO.from_grid(plasma_config.path_plasma)

    # compute the current field on the plasma surface and check that the sum of the current in the poloidal direction
    # is equal to the net poloidal current in vmec
    js = np.einsum("oijk,o->ijk", cws.current_op, current.get_phi_mn())
    assert np.abs(-js[..., 0].sum(1) * cws.dv -
                  vmec.net_poloidal_current).max() < 1e-7

    # check that the sum of the current in the toroidal direction is equal to the net toroidal current in vmec
    assert np.abs(js[..., 1].sum(0) * cws.du).max() < 1e-8

    # check the computation of the net currents
    from stellacode.surface.utils import get_net_current

    # compute the net current on the coil surface
    coil = cws.get_coil(current.get_phi_mn())
    net_pol_curr = get_net_current(coil, toroidal=False)
    assert np.max(np.abs(-net_pol_curr - vmec.net_poloidal_current)) < 1e-7

    # check that the net toroidal current is zero
    net_tor_curr = get_net_current(coil, toroidal=True)
    assert np.max(np.abs(net_tor_curr)) < 1e-8


def test_regcoil_with_pwc_no_current_at_bc():
    """
    Test the current conservation at the boundary of a rotated surface fitted to a plasma surface.
    The rotated surface has no current at the boundary.
    """
    # Define the number of Fourier coefficients for the current
    current_n_coeff = 8

    # Define the number of points in the integration grid
    n_points = current_n_coeff * 4

    # Initialize the EMCost object with the plasma configuration and integration parameters
    em_cost = EMCost.from_plasma_config(
        plasma_config=w7x_plasma,
        integration_par=IntegrationParams(
            num_points_u=n_points, num_points_v=n_points, center_vgrid=True),
        lamb=1e-14,
    )

    # Define the current operator with zero current at the boundary
    current = CurrentZeroTorBC(
        num_pol=current_n_coeff,
        num_tor=current_n_coeff,
        sin_basis=True,
        cos_basis=True,
        net_currents=get_net_current(w7x_plasma.path_plasma),
    )

    # Initialize an empty cylindrical surface
    fourier_coeffs = np.zeros((0, 2))
    surface = CylindricalSurface(
        fourier_coeffs=fourier_coeffs,
        integration_par=IntegrationParams(
            num_points_u=n_points, num_points_v=n_points, center_vgrid=True),
        nfp=9,
    )

    # Define a surface factory that first applies a cylindrical surface and then rotates the coil
    factory = Sequential(
        surface_factories=[
            surface,
            rotate_coil(
                current=current,
                nfp=3,
                num_surf_per_period=3,
                continuous_current_in_period=False,
            ),
        ]
    )

    # Fit the rotated surface to the plasma surface
    new_factory = fit_to_surface(factory, em_cost.Sp)
    new_factory.surface_factories[0].update_params(
        radius=new_factory.surface_factories[0].radius + 0.3)

    # Compute the current operator on the fitted surface
    new_surface = new_factory()
    phi_mn = em_cost.get_current_weights(new_surface)
    j_s = new_surface.get_j_surface(phi_mn)

    # Check that the current is zero at the boundary
    assert np.max(np.abs(j_s[:, 0, 1]) / np.max(np.abs(j_s))) < 0.1

    # Compute the current field in 3D
    j_3d = new_surface.get_j_3d(phi_mn)

    # Uncomment the following lines to plot the fitted surface and the current field
    # new_surface.plot(vector_field=j_3d)
    # new_surface.surface.plot(vector_field=j_3d, detach_parts=True)


def test_plot_plasma_cross_sections():
    """
    Test plotting of cross sections of a surface.

    This function creates a FourierSurfaceFactory object from a surface file
    and plots the cross sections of the surface.
    """

    # Create a FourierSurfaceFactory object from a surface file
    surf = FourierSurfaceFactory.from_file(
        ncsx_plasma.path_plasma,  # Path to the surface file
        integration_par=IntegrationParams(num_points_u=32, num_points_v=32),
        n_fp=3,  # Number of field periods
    )

    # Plot the cross sections of the surface
    # The plot is returned as a figure object and an axes object
    fig, ax = surf.plot_cross_sections(
        num_cyl=3,  # Number of cylinders to plot
        num=5,  # Number of cross sections to plot
        concave_envelope=True,  # Whether to plot the concave envelope
    )

    # Uncomment the following lines to display the plot
    # import matplotlib.pyplot as plt
    # plt.show()


def test_jac_hessian():
    """
    Test the jacobian and hessian calculations for a surface.

    This function creates a surface from a plasma configuration,
    calculates the jacobian and hessian of the surface, and
    checks that they are close to the analytical approximations.
    """

    # Create a surface from a plasma configuration
    cws = get_cws_from_plasma_config(w7x_plasma, n_harmonics_current=4)
    surf = cws()

    # Calculate the jacobian of the surface
    jac_approx = np.stack(
        np.gradient(surf.xyz, surf.du, surf.dv, axis=(0, 1)), axis=-1)

    # Check that the jacobian is close to the analytical approximation
    assert np.mean(np.abs(jac_approx - surf.jac_xyz)) / \
        np.mean(np.abs(surf.jac_xyz)) < 0.04, "Jacobian calculation failed"

    # Calculate the hessian of the surface
    hess_approx = np.stack(
        np.gradient(surf.jac_xyz, surf.du, surf.dv, axis=(0, 1)), axis=-1)

    # Check that the hessian is close to the analytical approximation
    assert np.mean(np.abs(hess_approx - surf.hess_xyz)) / \
        np.mean(np.abs(surf.hess_xyz)) < 0.08, "Hessian calculation failed"
