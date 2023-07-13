import typing as tp

import jax
from jax.typing import ArrayLike
from pydantic import BaseModel, Extra

from stellacode import np
import numpy as onp
from stellacode.tools.utils import get_min_dist

from .utils import get_principles


class IntegrationParams(BaseModel):
    num_points_u: int
    num_points_v: int
    max_val_v: float = 1.0

    @classmethod
    def from_current_potential(cls, current_pot):
        return cls(current_pot.num_pol * 4, current_pot.num_tor * 4)

    def get_uvgrid(self, concat: bool = False):
        u = np.linspace(0, 1, self.num_points_u, endpoint=False)
        v = np.linspace(0, self.max_val_v, self.num_points_v, endpoint=False)

        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")
        if concat:
            return np.stack((ugrid, vgrid), axis=0)
        else:
            return ugrid, vgrid

    @property
    def nbpts(self):
        return (self.num_points_u, self.num_points_v)

    @property
    def npts(self):
        return self.num_points_u * self.num_points_v

    @property
    def dudv(self):
        return self.max_val_v / (self.num_points_u * self.num_points_v)


class AbstractSurface(BaseModel):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    integration_par: IntegrationParams
    grids: tp.Optional[ArrayLike] = None
    num_tor_symmetry: int = 1
    trainable_params: tp.List[str] = []
    xyz: tp.Optional[ArrayLike] = None
    jac_xyz: tp.Optional[ArrayLike] = None
    normal: tp.Optional[ArrayLike] = None
    normal_unit: tp.Optional[ArrayLike] = None
    ds: tp.Optional[ArrayLike] = None
    principles: tp.Optional[tp.Tuple[ArrayLike, ArrayLike]] = None
    max_val_v: float = 1.0

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow  # allow extra fields

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_surface_attributes()

    @property
    def nbpts(self):
        return self.integration_par.nbpts

    @property
    def npts(self):
        return self.integration_par.npts

    @property
    def dudv(self):
        return self.integration_par.dudv

    def get_xyz(self, uv):
        """return the point parametrized by uv in cartesian coordinate"""
        raise NotImplementedError

    def get_trainable_params(self):
        return {k: getattr(self, k) for k in self.trainable_params}

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.compute_surface_attributes(deg=2)

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

    def compute_surface_attributes(self, deg: int = 2):
        """compute surface elements used in the shape optimization up
        to degree deg
        deg is 0,1 or 2"""
        self.grids = self.integration_par.get_uvgrid()
        uv_grid = np.stack(self.grids, axis=0)

        self.xyz = self.get_xyz_on_grid(uv_grid)

        # We also compute surface element dS and derivatives dS_u and dS_v:
        if deg >= 1:
            self.jac_xyz = self.get_jac_xyz_on_grid(uv_grid)

            N = np.cross(self.jac_xyz[0], self.jac_xyz[1], 0, 0, 0)
            self.normal = N
            self.ds = np.linalg.norm(N, axis=0)
            self.normal_unit = N / self.ds  # normal inward unit vector

        if deg >= 2:
            hess = self.get_hess_xyz_on_grid(uv_grid)
            self.principles = get_principles(hess_xyz=hess, jac_xyz=self.jac_xyz, normal_unit=self.normal_unit)

    def get_distance(self, xyz):
        return np.linalg.norm(self.xyz[..., None, None, :] - xyz[None, None, ...], axis=-1)

    def get_min_distance(self, xyz):
        return get_min_dist(self.xyz, xyz)

    def expand_for_plot_part(self):
        """Returns X, Y, Z arrays of one field period, adding redundancy of first column."""
        import numpy as np

        P = np.array(self.xyz)
        return [np.concatenate((P, P[:1]), axis=0)]

    def expand_for_plot_whole(self, detach_parts: bool = False):
        """Returns X, Y, Z arrays of the whole Stellarator."""
        import numpy as np

        points = self.expand_for_plot_part()[0]

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
        if detach_parts:
            return points_
        else:
            points = np.concatenate(points_, axis=1)
            return [np.concatenate((points, points[:, :1]), axis=1)]

    def plot(
        self,
        scalar: tp.Optional[onp.ndarray] = None,
        vector_field: tp.Optional[onp.ndarray] = None,
        only_one_period: bool = False,
        representation: str = "surface",
        color: tp.Optional[str] = None,
        colormap: str = "Wistia",
        detach_parts: bool = False,
    ):
        """Plot the surface"""
        import numpy as np
        from mayavi import mlab

        if only_one_period:
            xyz = self.expand_for_plot_part()
        else:
            xyz = self.expand_for_plot_whole(detach_parts)

        kwargs = {}
        if scalar is not None:
            kwargs["scalars"] = np.concatenate((scalar, scalar[0:1]), axis=0)

        for xyz_ in xyz:
            surf = mlab.mesh(
                xyz_[..., 0],
                xyz_[..., 1],
                xyz_[..., 2],
                representation=representation,
                colormap=colormap,
                color=color,
                **kwargs,
            )
        if vector_field is not None:
            vector_field = vector_field / np.max(vector_field)
            max_tor = xyz.shape[1]
            mlab.quiver3d(
                xyz[:-1, :, 0],
                xyz[:-1, :, 1],
                xyz[:-1, :, 2],
                vector_field[:, :max_tor, 0],
                vector_field[:, :max_tor, 1],
                vector_field[:, :max_tor, 3],
                line_width=0.5,
                scale_factor=0.3,
            )

        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100), np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100), np.linspace(0, 10, 100), color=(0, 0, 1))

        if scalar is not None:
            mlab.colorbar(surf, nb_labels=4, label_fmt="%.1E", orientation="vertical")

        mlab.view(39.35065816238082, 52.4711478893247, 4.259806307223165, np.array([4.10903064, 2.9532187, 4.20616949]))
