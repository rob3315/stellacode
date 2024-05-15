from os.path import dirname, join, realpath
import configparser

import numpy as np

from stellacode.costs.em_cost import EMCost
from stellacode.surface import (
    Current,
    CylindricalSurface,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.factory_tools import Sequential
from stellacode.surface.tore import ToroidalSurface


def test_compare_axisymmetric_vs_cylindrical():
    path_config = join(f"{(dirname(realpath(__file__)))}", "data","w7x","config.ini")
    # path_config = "test/data/w7x/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config)

    em_cost = EMCost.from_config(config, use_mu_0_factor=False)

    rotate_diff_current = 32
    major_radius = 5.5
    minor_radius = 1.036458468437195
    nfp = int(config["geometry"]["Np"])
    net_currents = np.array(
        [
            float(config["other"]["net_poloidal_current_Amperes"]) / nfp,
            float(config["other"]["net_toroidal_current_Amperes"]),
        ]
    )
    current = Current(num_pol=8, num_tor=8, net_currents=net_currents)
    n_pol_coil = 32
    n_tor_coil = 32
    total_num_rot = nfp * rotate_diff_current
    surface = CylindricalSurface(
        integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil // rotate_diff_current),
        nfp=total_num_rot,
        make_joints=False,
        distance=major_radius,
        radius=minor_radius,
        axis_angle=1.57079631 - (np.pi / 2 + np.pi / total_num_rot),
    )
    from stellacode.surface.factory_tools import rotate_coil

    surf_pwc = Sequential(
        surface_factories=[
            ToroidalSurface(
                nfp=total_num_rot,
                major_radius=major_radius,
                minor_radius=minor_radius,
                integration_par=IntegrationParams(
                    num_points_u=n_pol_coil, num_points_v=n_tor_coil // rotate_diff_current
                ),
            ),
            rotate_coil(
                current=current,
                nfp=nfp,
                num_surf_per_period=rotate_diff_current,
                continuous_current_in_period=True,
            ),
        ]
    )()

    surf_axi = Sequential(
        surface_factories=[
            ToroidalSurface(
                nfp=nfp,
                major_radius=major_radius,
                minor_radius=minor_radius,
                integration_par=IntegrationParams(num_points_u=n_pol_coil, num_points_v=n_tor_coil),
            ),
            rotate_coil(
                current=current,
                nfp=nfp,
                num_surf_per_period=1,
                continuous_current_in_period=False,
            ),
        ]
    )()

    # rtp_pwc = surf_pwc.cartesian_to_toroidal()
    # rtp_axi = surf_axi.cartesian_to_toroidal()
    # print(np.abs(surf_pwc.xyz - surf_axi.xyz).max() / np.abs(surf_pwc.xyz).mean())
    # print(np.abs(surf_pwc.jac_xyz[...,1] - surf_axi.jac_xyz[...,1]/32).max() / np.abs(surf_pwc.jac_xyz).mean())
    # print(np.abs(surf_pwc.ds - surf_axi.ds/32).max() / np.abs(surf_axi.ds).mean())
    # print(np.abs(surf_pwc.normal - surf_axi.normal).max() / np.abs(surf_axi.normal).mean())
    # print(np.abs(surf_pwc.normal_unit - surf_axi.normal_unit).max() / np.abs(surf_axi.normal_unit).mean())
    # print(np.abs(surf_pwc.npts - surf_axi.npts))

    np.testing.assert_allclose(em_cost.cost(surf_pwc)[1]["cost_B"], em_cost.cost(surf_axi)[1]["cost_B"])
