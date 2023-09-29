from stellacode.surface import CylindricalSurface, IntegrationParams
import numpy as np


def test_CylindricalSurface():
    cyl = CylindricalSurface(
        integration_par=IntegrationParams(num_points_u=8, num_points_v=8),
        num_tor_symmetry=3,
        axis_angle=0.3,
        make_joints=False,
    )
    rtz = cyl.to_cylindrical(cyl().xyz)
    assert np.allclose(rtz[:, :, 0], 1)
    assert np.allclose(rtz[:, :, 2] - rtz[:1, :, 2], 0.0)
