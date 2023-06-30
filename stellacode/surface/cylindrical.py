from jax.typing import ArrayLike

from stellacode import np

from .abstract_surface import AbstractSurface


class CylindricalSurface(AbstractSurface):
    Np: int
    axis_angle = 0.0
    radius = 1.0
    scale_length = 1.0
    distance = 3
    make_joints: bool = True

    def get_xyz(self, uv):
        u_ = 2 * np.pi * uv[0]  # poloidal variable
        v_ = (uv[1] - 0.5) * 2  # length variable

        _length = (
            self.scale_length
            * 2
            * (self.distance - self.radius)
            * np.sin(np.pi / self.Np)
        )

        axis_a = self.axis_angle
        axis_orth = np.array([np.sin(axis_a), -np.cos(axis_a), 0.0])

        z_dir = np.array([0.0, 0.0, 1.0])

        fourier_coeffs = self.params["fourier_coeffs"]
        angle = u_ * (np.arange(fourier_coeffs.shape[0]) + 1)

        _radius = (
            np.einsum("ab,ab",
                fourier_coeffs, np.stack((np.cos(angle), np.sin(angle)), axis=1),
            )
            + 1
        ) * self.radius

        circle = _radius * (axis_orth * np.cos(u_) + z_dir * np.sin(u_))

        if self.make_joints:
            p_dist = self.distance + np.cos(u_) * _radius
            _length = _length * p_dist / (self.distance - self.radius)

        cyl_axis = np.array(
            [np.cos(axis_a) * v_ * _length, np.sin(axis_a) * v_ * _length, 0.0]
        )

        return cyl_axis + circle + self.distance * axis_orth
