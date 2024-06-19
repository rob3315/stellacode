import typing as tp

import jax
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
from jax.typing import ArrayLike
from pydantic import BaseModel, Extra

import plotly.graph_objects as go

from stellacode import np
from stellacode.tools.rotate_n_times import RotateNTimes
from stellacode.surface.utils import get_principles, get_min_dist


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
    center_vgrid: bool = False

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
        if self.center_vgrid:
            v += self.du / 2
        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")
        return ugrid, vgrid

    @property
    def nbpts(self):
        """Number of points in the u and v directions"""
        return (self.num_points_u, self.num_points_v)

    @property
    def npts(self):
        """Number of points in the uv plane"""
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
    model_config = dict(arbitrary_types_allowed=True)

    def get_trainable_params(self):
        return {}

    def update_params(self, **kwargs):
        pass

    def __call__(self, surface=None, deg: int = 2, **kwargs):
        raise NotImplementedError

    def __gt__(self, surface):
        """Overload the > operator to build a pipe of surface factories."""
        from .factory_tools import Sequential

        if isinstance(self, Sequential):
            return self.model_copy(update=dict(surface_factories=self.surface_factories + [surface]))
        elif isinstance(surface, Sequential):
            return surface.model_copy(update=dict(surface_factories=[self] + surface.surface_factories))
        else:
            return Sequential(surface_factories=[self, surface])


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

    def get_uv_unwrapped(self, uv):
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

    def get_xyz_over_line(self, grid_):
        """Return the surface points over a line"""
        surf = jax.vmap(self.get_xyz, in_axes=1, out_axes=0)
        xyz = surf(grid_)
        # xyz = np.reshape(surf_res, (lu, lv, 3))

        return xyz

    def get_uv_unwrapped_on_grid(self, grid):
        """Return the surface points on the grid"""
        _, lu, lv = grid.shape
        grid_ = np.reshape(grid, (2, -1))
        surf = jax.vmap(self.get_uv_unwrapped, in_axes=1, out_axes=0)
        surf_res = surf(grid_)
        uv_unwrapped = np.reshape(surf_res, (lu, lv, 2))

        return uv_unwrapped

    def get_uv_unwrapped_over_line(self, grid_):
        """Return the surface points over a line"""
        surf = jax.vmap(self.get_uv_unwrapped, in_axes=1, out_axes=0)
        uv_unwrapped = surf(grid_)
        # xyz = np.reshape(surf_res, (lu, lv, 3))

        return uv_unwrapped

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
        hess_surf = jax.hessian(self.get_xyz, argnums=0)
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
            surface.normal = np.cross(
                surface.jac_xyz[..., 0], surface.jac_xyz[..., 1], -1, -1, -1)
            surface.ds = np.linalg.norm(surface.normal, axis=-1)
            surface.normal_unit = surface.normal / \
                surface.ds[:, :, None]  # normal inward unit vector

        if deg >= 2:
            surface.hess_xyz = self.get_hess_xyz_on_grid(uv_grid)

            surface.principle_max, surface.principle_min = get_principles(
                hess_xyz=surface.hess_xyz,
                jac_xyz=surface.jac_xyz,
                normal_unit=surface.normal_unit,
            )
            # surface.grad_ds = get_ds_grad(surface.jac_xyz, surface.hess_xyz)
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

    model_config = dict(arbitrary_types_allowed=True)

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
        # np.einsum(f"ij,ij{add_dims}->{add_dims}", self.ds, field) * self.dudv #
        return np.einsum(f"ij,{add_dims}ij->{add_dims}", self.ds, field) * self.dudv

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

    def expand_for_plot_part(self, num_tor_pts=1e11):
        """
        Returns X, Y, Z arrays of one field period, adding redundancy of the first column.
        :param num_tor_pts: Number of toroidal points to take from the surface.
        :return: A list of arrays of shape (N+1, 3), where N is the number of points
        taken from the surface.
        """
        import numpy as np

        num_pts = min(self.xyz.shape[1], num_tor_pts)

        # Take the first num_tor_pts points from the surface and concatenate with the first point
        P = np.array(self.xyz[:, :num_pts])
        return [np.concatenate((P, P[:1]), axis=0)]

    def expand_for_plot_whole(self, nfp: int, detach_parts: bool = False):
        """
        Returns X, Y, Z arrays of the whole Stellarator.

        Args:
            nfp (int): Number of field periods to include.
            detach_parts (bool, optional): If True, returns a list of arrays,
                where each array corresponds to a field period. If False, returns
                a single array containing all points. Defaults to False.

        Returns:
            Union[List[np.ndarray], np.ndarray]: Arrays or list of arrays of shape
                (N+1, 3), where N is the number of points in a field period,
                including the redundancy of the first column.
        """
        import numpy as np

        # Get the points for one field period
        points = self.expand_for_plot_part()[0]

        # Rotate the points to include all field periods
        points_rot = RotateNTimes.from_nfp(nfp)(points)

        # Reshape the rotated points to include all field periods
        pol, torrot, _ = points_rot.shape
        points_rot = np.reshape(points_rot, (pol, nfp, torrot // nfp, 3))

        # Split the rotated points into separate arrays for each field period
        points_ = [points_rot[:, i] for i in range(nfp)]

        if detach_parts:
            # Return the points for each field period as a list of arrays
            return points_
        else:
            # Concatenate the points for each field period into a single array
            points = np.concatenate(points_, axis=1)

            # Add the redundancy of the first column to the last point
            return [np.concatenate((points, points[:, :1]), axis=1)]

    def plot(
        self,
        scalar: tp.Optional[onp.ndarray] = None,
        vector_field: tp.Optional[onp.ndarray] = None,
        nfp: tp.Optional[int] = None,
        colormap: str = "Reds",
        detach_parts: bool = False,
        surface_kwargs: dict = dict(),
        cone_kwargs: dict = dict(
            sizeref=1,
            colorscale="Viridis",  # Define color of cones
        ),
        streamlines: bool = False,
        num_tor_pts: int = 1000000000000,
        reduce_res: int = 10,
        cut_tor: tp.Optional[int] = None,
        scale: bool = False,
    ):
        """
        Plot the surface, a scalar field or a vector field.

        Args:
            scalar (ndarray, optional): Scalar field to plot. Defaults to None.
            vector_field (ndarray, optional): Vector field to plot. Defaults to None.
            nfp (int, optional): Number of field periods. Defaults to None.
            colormap (str, optional): Colormap to use. Defaults to "Reds".
            detach_parts (bool, optional): Whether to detach parts. Defaults to False.
            surface_kwargs (dict, optional): Keyword arguments for surface. Defaults to dict().
            cone_kwargs (dict, optional): Keyword arguments for cone. Defaults to dict(sizeref=1, colorscale="Viridis").
            streamlines (bool, optional): Whether to use streamlines. Defaults to False.
            num_tor_pts (int, optional): Number of toroidal points. Defaults to 1000000000000.
            reduce_res (int, optional): Reduce resolution. Defaults to 10.
            cut_tor (int, optional): Cut toroidal direction. Defaults to None.
            scale (bool, optional): Whether to scale the vector field. Defaults to False.

        Returns:
            go.Figure: Plotly figure.
        """

        if nfp is None:
            xyz = self.expand_for_plot_part(num_tor_pts=num_tor_pts)
            reduce_res_nfp = 1
        else:
            xyz = self.expand_for_plot_whole(
                detach_parts=detach_parts, nfp=nfp)
            reduce_res_nfp = nfp

        if scalar is not None:
            scalar_ = onp.concatenate((scalar, scalar[0:1, :]), axis=0)
            scalar_ = onp.tile(scalar_, (1, reduce_res_nfp))
            if cut_tor is None:
                scalar_ = onp.concatenate((scalar_, scalar_[:, 0:1]), axis=1)[
                    :xyz[0].shape[0]:reduce_res, :xyz[0].shape[1]:reduce_res]
            else:
                scalar_ = scalar_[:xyz[0].shape[0]:reduce_res,
                                  :xyz[0].shape[1]:reduce_res]
        else:
            scalar_ = None

        fig = go.Figure()

        for xyz_ in xyz:
            if cut_tor is not None:
                cuts = onp.arange(xyz_.shape[1] + 1, step=cut_tor)
            else:
                cuts = [0, num_tor_pts]

            if scalar_ is not None:
                max_index_scalar = scalar_.shape[1]
                showscale = True
            else:
                showscale = False

            # Plot each segment
            for first, last in zip(cuts[:-1], cuts[1:]):
                xyz_c = xyz_[:, first:last]
                x_c, y_c, z_c = xyz_c[..., 0], xyz_c[..., 1], xyz_c[..., 2]

                if scalar_ is not None:
                    scalar_field = scalar_[:, first % max_index_scalar:(
                        last % max_index_scalar) + max_index_scalar * ((last % max_index_scalar) == 0)]
                else:
                    scalar_field = None

                # Plot the surface
                surface = go.Surface(
                    x=x_c,
                    y=y_c,
                    z=z_c,
                    surfacecolor=scalar_field,
                    colorscale=colormap,
                    showscale=showscale,
                    **surface_kwargs,
                )

                fig.add_trace(surface)

                # Plot the vector field
                if vector_field is not None:
                    if scale:
                        scaling_factor = onp.max(vector_field)
                    else:
                        scaling_factor = 1
                    vector_field_ = vector_field[:,
                                                 first:last] / scaling_factor
                    xyz_c = xyz_c[:-1]
                    if nfp is not None:
                        xyz_c = xyz_c[:, :-1]
                    x, y, z = xyz_c[..., 0], xyz_c[..., 1], xyz_c[..., 2]

                    if streamlines:
                        # Plot streamlines
                        fig.add_trace(
                            go.Streamtube(
                                x=x.flatten(),
                                y=y.flatten(),
                                z=z.flatten(),
                                u=vector_field_[..., 0].flatten(),
                                v=vector_field_[..., 1].flatten(),
                                w=vector_field_[..., 2].flatten(),
                                starts=dict(
                                    x=x[0, :].flatten(),
                                    y=y[0, :].flatten(),
                                    z=z[0, :].flatten(),
                                ),
                                visible=True,
                                **cone_kwargs,
                            )
                        )
                    else:
                        # Plot cones
                        fig.add_trace(
                            go.Cone(
                                x=x[::reduce_res, ::reduce_res *
                                    reduce_res_nfp].flatten(),
                                y=y[::reduce_res, ::reduce_res *
                                    reduce_res_nfp].flatten(),
                                z=z[::reduce_res, ::reduce_res *
                                    reduce_res_nfp].flatten(),
                                u=vector_field_[::reduce_res,
                                                ::reduce_res, 0].flatten(),
                                v=vector_field_[::reduce_res,
                                                ::reduce_res, 1].flatten(),
                                w=vector_field_[::reduce_res,
                                                ::reduce_res, 2].flatten(),
                                visible=True,
                                **cone_kwargs,
                            )
                        )

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig

    def plot_2d_field(self, field, num_prec=2, ax=None):
        """
        x is the toroidal direction (second dimension, also labelled v)
        y is the poloidal direction (first dimension, also labelled u)
        """
        assert field.shape[0] == self.xyz.shape[0]
        assert field.shape[1] == self.xyz.shape[1]

        field_norm = np.linalg.norm(field, axis=-1)
        # parfois ijua parfois ijau...
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


def get_ds_grad(surf):
    norm_dx_uv = np.linalg.norm(surf.jac_xyz, axis=-2)
    di_inv_dj_x = np.einsum("ijl,ijdl,ijdkl->ijkl", 1 /
                            norm_dx_uv, surf.jac_xyz, surf.hess_xyz)
    return np.einsum("ijkl,ijl->ijk", di_inv_dj_x, norm_dx_uv[..., ::-1])


def get_inv_ds_grad(surf):
    return -get_ds_grad(surf) / surf.ds[:, :, None] ** 2
