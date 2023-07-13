import configparser
import numpy as np
from stellacode.costs.em_cost import EMCost
from stellacode.surface import ToroidalSurface, RotatedSurface, CylindricalSurface, CurrentPotential
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.tore import ToroidalSurface
import configparser
from stellacode.costs.em_cost import EMCost
from stellacode.surface import IntegrationParams


def test_compare_axisymmetric_toroidal():
    path_config = "test/data/w7x/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config)

    em_cost = EMCost.from_config(config, use_mu_0_factor=False)

    rotate_diff_current = 32
    major_radius = 5.5
    minor_radius = 1.036458468437195
    current = CurrentPotential(num_pol=8, num_tor=8)
    n_pol_coil = 32
    n_tor_coil = 32

    surf_pwc = RotatedSurface(
        surface=CylindricalSurface(
            integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil // rotate_diff_current),
            num_tor_symmetry=em_cost.Sp.num_tor_symmetry * rotate_diff_current,
            make_joints=False,
            distance=major_radius,
            radius=minor_radius,
            axis_angle=1.57079631,
        ),
        num_tor_symmetry=em_cost.Sp.num_tor_symmetry,
        rotate_diff_current=rotate_diff_current,
        current=current,
        common_current_on_each_rot=True,
    )

    tor_surf = ToroidalSurface(
        num_tor_symmetry=em_cost.Sp.num_tor_symmetry,
        major_radius=major_radius,
        minor_radius=minor_radius,
        integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil),
    )
    surf_axi = RotatedSurface(
        surface=tor_surf,
        num_tor_symmetry=em_cost.Sp.num_tor_symmetry,
        rotate_diff_current=1,
        current=current,
    )

    # rtp_pwc = surf_pwc.cartesian_to_toroidal()
    # rtp_axi = surf_axi.cartesian_to_toroidal()
    # print(np.abs(surf_pwc.xyz - surf_axi.xyz).max() / np.abs(surf_pwc.xyz).mean())
    # print(np.abs(surf_pwc.jac_xyz - surf_axi.jac_xyz).max() / np.abs(surf_pwc.jac_xyz).mean())
    # print(np.abs(surf_pwc.ds - surf_axi.ds).max() / np.abs(surf_axi.ds).mean())
    # print(np.abs(surf_pwc.normal - surf_axi.normal).max() / np.abs(surf_axi.normal).mean())
    # print(np.abs(surf_pwc.normal_unit - surf_axi.normal_unit).max() / np.abs(surf_axi.normal_unit).mean())
    # print(np.abs(surf_pwc.npts - surf_axi.npts))

    assert np.abs(em_cost.cost(surf_pwc)[1]["cost_B"] - em_cost.cost(surf_axi)[1]["cost_B"]) < 1e-11
