import typing as tp

import jax
import numpy as onp
from jax.typing import ArrayLike
from pydantic import BaseModel, Extra

from stellacode import np
from stellacode.tools.rotate_n_times import RotateNTimes
from stellacode.tools.utils import get_min_dist
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import get_principles


class IntegrationParams(BaseModel):
    """
    Represent a 2D grid for integration.

    Args:
        * num_points_u: number of points in the poloidal, u direction
        * num_points_v: number of points in the toroidal, v direction
        * max_val_v: maximum value in the toroidal, v direction
    """

    num_points_u: int
    num_points_v: int
    max_val_u: float = 1.0
    max_val_v: float = 1.0

    @classmethod
    def from_current_potential(cls, current_pot):
        return cls(current_pot.num_pol * 4, current_pot.num_tor * 4)

    @classmethod
    def sum_toroidally(cls, int_pars: list):
        nptu = {par.num_points_u for par in int_pars}
        assert len(nptu) == 1
        maxu = {par.max_val_u for par in int_pars}
        assert len(maxu) == 1
        nptv = sum([par.num_points_v for par in int_pars])
        maxv = {par.max_val_v for par in int_pars}
        assert len(maxv) == 1
        return cls(
            num_points_u=int_pars[0].num_points_u,
            num_points_v=nptv,
            max_val_u=int_pars[0].max_val_u,
            max_val_v=sum([par.max_val_v for par in int_pars]),
        )

    def get_uvgrid(self):
        u = np.linspace(0, self.max_val_u, self.num_points_u, endpoint=False)
        v = np.linspace(0, self.max_val_v, self.num_points_v, endpoint=False)

        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")
        return ugrid, vgrid

    @property
    def nbpts(self):
        return (self.num_points_u, self.num_points_v)

    @property
    def npts(self):
        return self.num_points_u * self.num_points_v

    @property
    def du(self):
        """Integration surface element in the poloidal, u direction"""
        return self.max_val_u / self.num_points_u

    @property
    def dv(self):
        """Integration surface element in the toroidal, v direction"""
        return self.max_val_v / self.num_points_v

    @property
    def dudv(self):
        """2D integration surface element"""
        return self.du * self.dv


class AbstractBaseFactory(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid

    def get_trainable_params(self):
        return {}

    def update_params(self, **kwargs):
        pass

    def __call__(self, surface=None, deg: int = 2):
        raise NotImplementedError


class AbstractSurfaceFactory(AbstractBaseFactory):
    """
    Abstract class for constructing surfaces

    Args:
        * integration_par: surface integration parameters
    """
    integration_par: IntegrationParams

    def get_xyz(self, uv):
        """return the surface point parametrized by uv in cartesian coordinate"""
        raise NotImplementedError

    def get_trainable_params(self):
        """Return the trainable parameters from the factory"""
        return {k: getattr(self, k) for k in self.trainable_params}

    def update_params(self, **kwargs):
        """Update the trainable parameters from the factory"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_xyz_on_grid(self, grid):
        """Return the surface points on the grid"""
        _, lu, lv = grid.shape
        grid_ = np.reshape(grid, (2, -1))
        surf = jax.vmap(self.get_xyz, in_axes=1, out_axes=0)
        surf_res = surf(grid_)
        xyz = np.reshape(surf_res, (lu, lv, 3))

        return xyz

    def get_jac_xyz_on_grid(self, grid):
        """Return the surface jacobian on the grid"""        
        _, lu, lv = grid.shape
        grid_ = np.reshape(grid, (2, -1))
        jac_surf = jax.jacobian(self.get_xyz, argnums=0)
        jac_surf_vmap = jax.vmap(jac_surf, in_axes=1, out_axes=0)
        jac_surf_res = jac_surf_vmap(grid_)
        jac_xyz = np.reshape(jac_surf_res, (lu, lv, 3, 2))

        return jac_xyz

    def get_hess_xyz_on_grid(self, grid):
        """Return the surface hessian on the grid"""          
        _, lu, lv = grid.shape
        grid_ = np.reshape(grid, (2, -1))
        hess_surf = jax.hessian(self.get_xyz, argnums=0, holomorphic=False)
        hess_surf_vmap = jax.vmap(hess_surf, in_axes=1, out_axes=0)
        hess_surf_res = hess_surf_vmap(grid_)

        return np.reshape(hess_surf_res, (lu, lv, 3, 2, 2))

    def __call__(self, deg: int = 2):
        """Compute a surface

        Args:   
            * deg: degree of elements computed
        """
        grids = self.integration_par.get_uvgrid()
        uv_grid = np.stack(grids, axis=0)
        surface = Surface(
            integration_par=self.integration_par,
            grids=grids,
            xyz=self.get_xyz_on_grid(uv_grid),
        )

        surface.integration_par = self.integration_par
        surface.grids = grids

        # We also compute surface element dS and derivatives dS_u and dS_v:
        if deg >= 1:
            surface.jac_xyz = self.get_jac_xyz_on_grid(uv_grid)
            surface.normal = np.cross(surface.jac_xyz[..., 0], surface.jac_xyz[..., 1], -1, -1, -1)
            surface.ds = np.linalg.norm(surface.normal, axis=-1)
            surface.normal_unit = surface.normal / surface.ds[:, :, None]  # normal inward unit vector

        if deg >= 2:
            surface.hess_xyz = self.get_hess_xyz_on_grid(uv_grid)

            surface.principle_max, surface.principle_min = get_principles(
                hess_xyz=surface.hess_xyz,
                jac_xyz=surface.jac_xyz,
                normal_unit=surface.normal_unit,
            )
        return surface


class Surface(BaseModel):
    """Represent a surface

    Args:
        * integration_par: parameters for the 2D grid
        * grids: 2D grids in cartesian coordinates
        * xyz: surface points in cartesian coordinates
        * jac_xyz: surface jacobian in cartesian coordinates
        * hess_xyz: surface hessian in cartesian coordinates
        * normal: surface normal vector pointing inside the surface
        * normal_unit: normalized surface normal vector pointing inside the surface
        * ds: surface covered by each grid point
        * pinciple max: 
        * principle_min: 
    """

    integration_par: IntegrationParams
    grids: tp.Tuple[ArrayLike, ArrayLike]
    xyz: ArrayLike
    jac_xyz: tp.Optional[ArrayLike] = None
    hess_xyz: tp.Optional[ArrayLike] = None
    normal: tp.Optional[ArrayLike] = None
    normal_unit: tp.Optional[ArrayLike] = None
    ds: tp.Optional[ArrayLike] = None
    principle_max: tp.Optional[ArrayLike] = None
    principle_min: tp.Optional[ArrayLike] = None

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shu = set()
        shv = set()
        for k in self.field_keys:
            if getattr(self, k) is not None:
                shape = getattr(self, k).shape
                shu.add(shape[0])
                shv.add(shape[1])
                assert len(shu) in [0, 1]
                assert len(shv) in [0, 1]
        if self.grids is not None:
            for grid in self.grids:
                shu.add(grid.shape[0])
                shv.add(grid.shape[1])
                assert len(shu) in [0, 1]
                assert len(shv) in [0, 1]
        if self.integration_par is not None:
            shu.add(self.integration_par.num_points_u)
            shv.add(self.integration_par.num_points_v)
            assert len(shu) in [0, 1]
            assert len(shv) in [0, 1]

    @property
    def field_keys(self):
        """
        Keys that represent fields on the surface: the first two dimensions are respectively the 
        poloidal and toroidal dimensions.
        """
        return ["xyz", "jac_xyz", "hess_xyz", "normal", "normal_unit", "ds", "principle_max", "principle_min"]

    @property
    def nbpts(self):
        return self.integration_par.nbpts

    @property
    def npts(self):
        return self.integration_par.npts

    @property
    def du(self):
        return self.integration_par.du

    @property
    def dv(self):
        return self.integration_par.dv

    @property
    def dudv(self):
        return self.integration_par.dudv

    def get_distance(self, xyz):
        return np.linalg.norm(self.xyz[..., None, None, :] - xyz[None, None, ...], axis=-1)

    def get_min_distance(self, xyz):
        return get_min_dist(self.xyz, xyz)

    def integrate(self, field):
        add_dims = "abcd"[: len(field.shape) - 2]
        return np.einsum(f"ij,ij{add_dims}->{add_dims}", self.ds, field) * self.dudv * self.nfp

    @property
    def area(self):
        return np.sum(self.ds) * self.dudv

    def get_g_lower_covariant(self):
        """
        Covariant surface metric g_{ij}
        """
        return np.einsum("ijab,ijac->ijbc", self.jac_xyz, self.jac_xyz)

    def get_g_upper_contravariant(self):
        """
        Contravariant surface metric g^{ij}
        """
        return np.linalg.pinv(self.get_g_lower_covariant())

    def get_g_upper_basis(self):
        """
        Return the contravariant basis vectors
        """
        g_up = self.get_g_upper_contravariant()
        return np.einsum("...kl,...ak->...al", g_up, self.jac_xyz)

    def expand_for_plot_part(self, num_tor_pts=100000000000):
        """Returns X, Y, Z arrays of one field period, adding redundancy of first column."""
        import numpy as np

        P = np.array(self.xyz[:, :num_tor_pts])
        return [np.concatenate((P, P[:1]), axis=0)]

    def expand_for_plot_whole(self, nfp: int, detach_parts: bool = False):
        """Returns X, Y, Z arrays of the whole Stellarator."""
        import numpy as np

        points = self.expand_for_plot_part()[0]

        points_rot = RotateNTimes.from_nfp(nfp)(points)
        pol, torrot, _ = points_rot.shape
        points_rot = np.reshape(points_rot, (pol, nfp, torrot // nfp, 3))
        points_ = [points_rot[:, i] for i in range(nfp)]

        if detach_parts:
            return points_
        else:
            points = np.concatenate(points_, axis=1)
            return [np.concatenate((points, points[:, :1]), axis=1)]

    def plot(
        self,
        scalar: tp.Optional[onp.ndarray] = None,
        vector_field: tp.Optional[onp.ndarray] = None,
        nfp: tp.Optional[int] = None,
        representation: str = "surface",
        color: tp.Optional[str] = None,
        colormap: str = "Wistia",
        detach_parts: bool = False,
        quiver_kwargs: dict = dict(
            line_width=0.5,
            scale_factor=0.1,
        ),
        mesh_kwargs: dict = dict(),
        num_tor_pts: int = 1000000000000,
        reduce_res: int = 1,
        cut_tor: tp.Optional[int]=None,
    ):
        """Plot the surface"""
        import numpy as np
        from mayavi import mlab

        if nfp is None:
            xyz = self.expand_for_plot_part(num_tor_pts=num_tor_pts)
        else:
            xyz = self.expand_for_plot_whole(detach_parts, nfp=nfp)

        kwargs = mesh_kwargs
        if scalar is not None:
            scalar_ = np.concatenate((scalar, scalar[0:1]), axis=0)
            kwargs["scalars"] = np.concatenate((scalar_, scalar_[:, 0:1]), axis=1)[::reduce_res, ::reduce_res]

        for xyz_ in xyz:
            if cut_tor is not None:
                cuts = np.arange(xyz_.shape[1] + 1, step=cut_tor)
            else:
                cuts = [0, 100000000]
            for first, last in zip(cuts[:-1], cuts[1:]):
                xyz_c = xyz_[:, first:last]
                surf = mlab.mesh(
                    xyz_c[..., 0],
                    xyz_c[..., 1],
                    xyz_c[..., 2],
                    representation=representation,
                    colormap=colormap,
                    color=color,
                    **kwargs,
                )
                if vector_field is not None:
                    vector_field_ = vector_field[:, first:last] / np.max(vector_field)

                    xyz_c = xyz_c[:-1]
                    mlab.quiver3d(
                        xyz_c[::reduce_res, ::reduce_res, 0],
                        xyz_c[::reduce_res, ::reduce_res, 1],
                        xyz_c[::reduce_res, ::reduce_res, 2],
                        vector_field_[::reduce_res, ::reduce_res, 0],
                        vector_field_[::reduce_res, ::reduce_res, 1],
                        vector_field_[::reduce_res, ::reduce_res, 2],
                        **quiver_kwargs,
                    )

        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100), np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100), np.linspace(0, 10, 100), color=(0, 0, 1))

        if scalar is not None:
            mlab.colorbar(surf, nb_labels=4, label_fmt="%.1E", orientation="vertical")

        mlab.view(39.35065816238082, 52.4711478893247, 4.259806307223165, np.array([4.10903064, 2.9532187, 4.20616949]))

    def plot_2d_field(self, field, num_prec=2, ax=None):
        """
        x is the toroidal direction (second dimension, also labelled v)
        y is the poloidal direction (first dimension, also labelled u)
        """
        assert field.shape[0] == self.xyz.shape[0]
        assert field.shape[1] == self.xyz.shape[1]

        field_norm = np.linalg.norm(field, axis=-1)
        field_cov = np.einsum("ija,ijau->iju", field, self.jac_xyz)
        ax = sns.heatmap(field_norm, cmap="winter", ax=ax)

        x_pos = self.grids[1][::num_prec, ::num_prec].T * field_norm.shape[1]
        y_pos = self.grids[0][::num_prec, ::num_prec].T * field_norm.shape[0]
        x_field = field_cov[::num_prec, ::num_prec, 1].T
        y_field = field_cov[::num_prec, ::num_prec, 0].T
        ax.quiver(x_pos, y_pos, x_field, y_field, color="w", units="width")
        plt.ylabel("Poloidal angle")
        plt.xlabel("Toroidal angle")
        return ax
