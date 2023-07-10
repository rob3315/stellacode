import typing as tp

import jax
from jax.typing import ArrayLike
from pydantic import BaseModel, Extra

from stellacode import np
from stellacode.tools.utils import get_min_dist
from .utils import get_principles


class SurfaceAttributes(BaseModel):
    npts: int
    xyz: ArrayLike
    jac_xyz: tp.Optional[ArrayLike] = None
    normal: tp.Optional[ArrayLike] = None
    normal_unit: tp.Optional[ArrayLike] = None
    principles: tp.Optional[tp.Tuple[ArrayLike, ArrayLike]] = None

    class Config:
        arbitrary_types_allowed = True


class AbstractSurface(BaseModel):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    nbpts: tp.Tuple[int, int]
    num_tor_symmetry: int = 1
    trainable_params: tp.List[str] = []

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow  # allow extra fields

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.npts = self.nbpts[0] * self.nbpts[1]
        self.grids = self.get_uvgrid(*self.nbpts)
        self.compute_surface_attributes()  # computation of the surface attributes

    def get_xyz(self, uv):
        """return the point parametrized by uv in cartesian coordinate"""
        raise NotImplementedError

    def get_trainable_params(self):
        return {k: getattr(self, k) for k in self.trainable_params}

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.compute_surface_attributes(deg=2)

    @staticmethod
    def get_uvgrid(lu, lv, concat: bool = False):
        u, v = np.linspace(0, 1, lu, endpoint=False), np.linspace(0, 1, lv, endpoint=False)

        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")
        if concat:
            return np.stack((ugrid, vgrid), axis=0)
        else:
            return ugrid, vgrid

    def get_xyz_on_grid(self, grid):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape
        surf = jax.vmap(self.get_xyz, in_axes=1, out_axes=-1)
        surf_res = surf(grid_)
        lu, lv = self.nbpts
        xyz = np.transpose(np.reshape(surf_res, (3, lu, lv)), (1, 2, 0))
        return xyz

    def get_jac_xyz_on_grid(self, grid):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        jac_surf = jax.jacobian(self.get_xyz, argnums=0)
        jac_surf_vmap = jax.vmap(jac_surf, in_axes=1, out_axes=-1)
        jac_surf_res = jac_surf_vmap(grid_)
        jac_xyz = np.reshape(np.transpose(jac_surf_res, (1, 0, 2)), (2, 3, lu, lv))
        return jac_xyz

    def get_hess_xyz_on_grid(self, grid):
        grid_ = np.reshape(grid, (2, -1))
        _, lu, lv = grid.shape

        hess_surf = jax.hessian(self.get_xyz, argnums=0, holomorphic=False)
        hess_surf_vmap = jax.vmap(hess_surf, in_axes=1, out_axes=-1)
        hess_surf_res = hess_surf_vmap(grid_)
        return np.reshape(hess_surf_res, (3, 2, 2, lu, lv))

    def compute_surface_attributes(self, deg=2):
        """compute surface elements used in the shape optimization up
        to degree deg
        deg is 0,1 or 2"""

        uv_grid = np.stack(self.grids, axis=0)

        self.P = self.get_xyz_on_grid(uv_grid)

        # We also compute surface element dS and derivatives dS_u and dS_v:
        if deg >= 1:
            self.dpsi = self.get_jac_xyz_on_grid(uv_grid)

            N = np.cross(self.dpsi[0], self.dpsi[1], 0, 0, 0)
            self.N = N
            self.dS = np.linalg.norm(N, axis=0)
            self.n = N / self.dS  # normal inward unit vector

        if deg >= 2:
            hess = self.get_hess_xyz_on_grid(uv_grid)
            self.principles = get_principles(hess_xyz=hess, jac_xyz=self.dpsi, normal_unit=self.n)

    def get_distance(self, xyz):
        return np.linalg.norm(self.P[..., None, None, :] - xyz[None, None, ...], axis=-1)

    def get_min_distance(self, xyz):
        return get_min_dist(self.P, xyz)

    def expand_for_plot_part(self):
        """Returns X, Y, Z arrays of one field period, adding redundancy of first column."""
        import numpy as np

        P = np.array(self.P)
        return np.concatenate((P, P[:1]), axis=0)

    def expand_for_plot_whole(self):
        """Returns X, Y, Z arrays of the whole Stellarator."""
        import numpy as np

        points = self.expand_for_plot_part()

        points_ = [points]
        for i in range(1, self.num_tor_symmetry):
            angle = 2 * i * np.pi / self.num_tor_symmetry
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            points_.append(np.einsum("ij,uvj->uvi", rotation_matrix, points))
        points = np.concatenate(points_, axis=1)
        return np.concatenate((points, points[:, :1]), axis=1)

    def plot(
        self,
        scalar=None,
        only_one_period: bool = False,
        representation: str = "surface",
        color: tp.Optional[str] = None,
        colormap: str = "Wistia",
    ):
        """Plot the surface"""
        import numpy as np
        from mayavi import mlab

        if only_one_period:
            xyz = self.expand_for_plot_part()
        else:
            xyz = self.expand_for_plot_whole()

        kwargs = {}
        if scalar is not None:
            kwargs["scalars"] = np.concatenate((scalar, scalar[0:1]), axis=0)

        surf = mlab.mesh(
            xyz[..., 0],
            xyz[..., 1],
            xyz[..., 2],
            representation=representation,
            colormap=colormap,
            color=color,
            **kwargs,
        )
        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100), np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100), np.linspace(0, 10, 100), color=(0, 0, 1))
        if scalar is not None:
            mlab.colorbar(surf, nb_labels=4, label_fmt="%.1E", orientation="vertical")
