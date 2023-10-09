import jax

jax.config.update("jax_enable_x64", True)
import pytest
from stellacode.definitions import w7x_plasma, ncsx_plasma
from stellacode.surface import (
    Current,
    CurrentZeroTorBC,
    IntegrationParams,
    ToroidalSurface,
)
from stellacode.surface import CylindricalSurface, FourierSurfaceFactory
from stellacode.surface.imports import (
    get_cws_from_plasma_config,
)

import pytest
import numpy as onp


@pytest.mark.skip("plot")
def test_plot_current():
    cws = get_cws_from_plasma_config(w7x_plasma, n_harmonics_current=4)
    cws.current.set_phi_mn(onp.random.random(cws.current.phi_mn.shape) * 1e1)
    j_3d = cws.get_j_3d()
    cws.surface.plot(colormap="Blues", vector_field=j_3d)
    cws.plot_j_surface()
    # plt.show()


@pytest.mark.skip("plot")
def test_plot_b_field():
    surf = FourierSurfaceFactory.from_file(
        ncsx_plasma.path_plasma,
        integration_par=IntegrationParams(num_points_u=32, num_points_v=31),
        n_fp=3,
    )

    b_field = surf.get_gt_b_field()
    surf.plot(colormap="Blues", vector_field=b_field)
    surf.plot_2d_field(b_field[:, :31])
    # plt.show()
