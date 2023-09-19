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


def test_double_torus():
    current = Current(num_pol=8, num_tor=8, net_currents=get_net_current(w7x_plasma.path_plasma))

    cws = RotatedSurface(
        surface=ToroidalSurface(
            num_tor_symmetry=5,
            major_radius=5,
            minor_radius=1,
            params={},
            integration_par=current.get_integration_params(),
        ),
        num_tor_symmetry=5,
        rotate_diff_current=1,
        current=current,
    )
    from stellacode.surface.coil_surface import CoilSurface

    class DoubleToroidalSurface(CoilSurface):
        coil1: CoilSurface
        coil2: CoilSurface

        def compute_surface_attributes(self, deg=2):
            """compute surface elements used in the shape optimization up
            to degree deg
            deg is 0,1 or 2"""

            self.grids = self.coil1.grids
            self.coil1.compute_surface_attributes(deg=deg)
            for k in [
                "xyz",
                "jac_xyz",
                "normal",
                "ds",
                "normal_unit",
                "hess_xyz",
                "principle_max",
                "principle_min",
            ]:
                setattr(self.coil2.surface, k, getattr(self.coil1.surface, k))
            self.coil2.compute_surface_attributes(deg=deg)
            for k in [
                "major_radius",
                # "minor_radius",
                "fourier_coeffs",
            ]:
                val = getattr(self.coil1, k)
                val2 = getattr(self.coil2, k)
                if val is not None:
                    setattr(self, k, np.concatenate([val, val2], axis=1))

            self.coil1._set_curent_op()
            self.coil2._set_curent_op()

            self.current_op = np.concatenate([self.coil1.current_op, self.coil2.current_op], axis=2)

    DoubleToroidalSurface()

    em_cost = EMCost.from_plasma_config(
        plasma_config=w7x_plasma, integration_par=IntegrationParams(num_points_u=32, num_points_v=32)
    )
