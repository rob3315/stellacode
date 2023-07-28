import collections

import stellacode.tools as tools
from stellacode import np
from .utils import cartesian_to_toroidal
from .coil_surface import CoilSurface
from stellacode.tools.rotate_n_times import RotateNTimes


class RotatedSurface(CoilSurface):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    num_tor_symmetry: int = 1
    rotate_diff_current: int = 1
    common_current_on_each_rot: bool = False
    rotate_n: RotateNTimes = RotateNTimes(1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rotate_n = RotateNTimes(self.num_tor_symmetry * self.rotate_diff_current)
        self.compute_surface_attributes()
        assert self.num_tor_symmetry * self.rotate_diff_current == self.surface.num_tor_symmetry

    @property
    def dudv(self):
        if self.common_current_on_each_rot:
            return self.surface.dudv / self.rotate_diff_current
        else:
            return self.surface.dudv

    @property
    def area(self):
        return self.surface.area

    def get_num_rotations(self):
        return self.num_tor_symmetry * self.rotate_diff_current

    def get_trainable_params(self):
        return self.surface.get_trainable_params()

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.surface, k, v)
        self.compute_surface_attributes(deg=2)

    def get_curent_op(self):
        if self.common_current_on_each_rot:
            gridu, gridv = self.grids
            gridu = np.concatenate([gridu] * self.rotate_diff_current, axis=1)
            rd = self.rotate_diff_current
            # this scaling is necessary because the Current class expect the grid to
            # always be from 0 to 1.
            gridv = np.concatenate([(i + gridv) / rd for i in range(rd)], axis=1)
            blocks = self.current.get_matrix_from_grid((gridu, gridv))
            # this is because the current potential derivative vs v is scaled by rd
            # when v is scaled by 1/rd
            blocks[..., -1] *= rd

        else:
            curent_op = super().get_curent_op()
            current_op_ = curent_op[2:]

            inner_blocks = collections.deque(
                [current_op_] + [np.zeros_like(current_op_)] * (self.rotate_diff_current - 1)
            )
            blocks = []
            for _ in range(len(inner_blocks)):
                blocks.append(np.concatenate(inner_blocks, axis=0))
                inner_blocks.rotate(1)

            # This is a hack because the status of the first two coefficients is
            # special (constant currents not regressed)
            blocks = np.concatenate(blocks, axis=2)
            blocks = np.concatenate((np.concatenate([curent_op[:2]] * len(inner_blocks), axis=2), blocks), axis=0)

        return np.concatenate([blocks] * self.num_tor_symmetry, axis=2)

    def compute_surface_attributes(self, deg=2):
        """compute surface elements used in the shape optimization up
        to degree deg
        deg is 0,1 or 2"""

        self.grids = self.surface.grids
        self.surface.compute_surface_attributes(deg=deg)

        for k in ["xyz", "jac_xyz", "normal", "ds", "normal_unit", "hess_xyz"]:
            val = getattr(self.surface, k)
            if val is not None:
                setattr(self, k, self.rotate_n(val))

        if deg >= 2:
            self.principles = [self.rotate_n(val) for val in self.surface.principles]

        self.current_op = self.get_curent_op()

    def cartesian_to_toroidal(self):
        try:
            major_radius = self.surface.distance
        except:
            major_radius = self.surface.major_radius
        return cartesian_to_toroidal(xyz=self.xyz, tore_radius=major_radius, height=0.0)
