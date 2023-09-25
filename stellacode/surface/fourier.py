import typing as tp
from os import sep

import numpy as onp
from concave_hull import concave_hull
from jax.typing import ArrayLike
from scipy.interpolate import CubicSpline, interp1d
from scipy.io import netcdf_file
from scipy.spatial import ConvexHull

from stellacode import np
from stellacode.surface.utils import fourier_coefficients
from stellacode.tools.vmec import VMECIO

from .abstract_surface import AbstractSurfaceFactory, IntegrationParams, Surface
from .cylindrical import CylindricalSurface
from .tore import ToroidalSurface
from .utils import (
    cartesian_to_cylindrical,
    cartesian_to_shifted_cylindrical,
    cartesian_to_toroidal,
    from_polar,
    to_polar,
)
from stellacode.tools.bnorm import get_bnorm


class FourierSurface(AbstractSurfaceFactory):
    """A class used to represent a toroidal surface with Fourier coefficients

    :param params: (m,n,Rmn,Zmn) 4 lists to parametrize the surface
    :type params: (int[],int[],float[],float[])
    :param nbpts: see :func:`.abstract_surface.Abstract_surface`
    :type nbpts: (int,int)
    :param Np: see `.abstract_surface.Abstract_surface`
    :type Np: int
    """

    mf: ArrayLike
    nf: ArrayLike
    Rmn: ArrayLike
    Zmn: ArrayLike
    file_path: str

    trainable_params: tp.List[str] = [
        "Rmn",
        "Zmn",
    ]

    @classmethod
    def from_file(
        cls,
        path_surf: str,
        integration_par: IntegrationParams,
        n_fp=1,
        surface_label: int = -1,
    ):
        """This function returns a Surface_Fourier object defined by a file.
        Three kinds of file are currently supported :
        - wout nc files generated by VMEC (for plasma surfaces)
        - nescin files generated by regcoil
        - text files (see example in data/li383)

        load file with the format m,n,Rmn,Zmn"""

        if path_surf[-3::] == ".nc":
            vmec = VMECIO.from_grid(path_surf)
            m = vmec.get_var("xm", int)
            n = -vmec.get_var("xn", int) / vmec.nfp
            Rmn = vmec.get_var("rmnc", float)[surface_label]
            Zmn = vmec.get_var("zmns", float)[surface_label]
            n_fp = vmec.nfp

        elif path_surf.rpartition(sep)[-1][:6:] == "nescin":
            with open(path_surf, "r") as f:
                line = f.readline()
                while "------ Current Surface:" not in line:
                    line = f.readline()
                line = f.readline()
                num_modes = int(f.readline())

                f.readline()
                f.readline()

                data = []
                for _ in range(num_modes):
                    data.append(str.split(f.readline()))
                adata = np.array(data, dtype="float64")
                m, n, Rmn, Zmn = adata[:, 0], adata[:, 1], adata[:, 2], adata[:, 3]

        else:
            data = []
            with open(path_surf, "r") as f:
                next(f)
                for line in f:
                    data.append(str.split(line))

            adata = np.array(data, dtype="float64")
            m, n, Rmn, Zmn = adata[:, 0], adata[:, 1], adata[:, 2], adata[:, 3]

        return cls(
            Rmn=Rmn,
            Zmn=Zmn,
            mf=m,
            nf=n,
            integration_par=integration_par,
            num_tor_symmetry=n_fp,
            file_path=path_surf,
        )

    def get_xyz(self, uv):
        angle = 2 * np.pi * (uv[0] * self.mf + uv[1] * self.nf)
        R = np.tensordot(self.Rmn, np.cos(angle), 1)
        Z = np.tensordot(self.Zmn, np.sin(angle), 1)
        phi = 2 * np.pi * uv[1] / self.num_tor_symmetry
        return np.array([R * np.cos(phi), R * np.sin(phi), Z])

    def get_major_radius(self):
        assert self.mf[0] == 0 and self.nf[0] == 0
        return self.Rmn[0]

    def plot_cross_sections(
        self,
        num_cyl: tp.Optional[int] = None,
        num: int = 5,
        convex_envelope: bool = True,
        concave_envelope: bool = False,
        scale_envelope: float = 1.0,
        ax=None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        u = np.linspace(0, 1, 100, endpoint=True)
        v = np.linspace(0, 1, num, endpoint=True)
        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")
        surf = self()
        xyz = self.get_xyz_on_grid(np.stack((ugrid, vgrid)))

        rtheta = surf._get_rtheta(xyz=xyz)

        for i in range(num):
            rphi = np.concatenate((rtheta[:, i, :], rtheta[:, i, :]), axis=0)
            ax.plot(rphi[:, 1], rphi[:, 0], c=[0, 0] + [float((i + 1) / (num + 1))])

        if convex_envelope:
            env = surf.get_envelope(num_cyl=num_cyl, convex=True)
            ax.plot(env[:, 1], env[:, 0] * scale_envelope, c="r", linewidth=3)
        if concave_envelope:
            env = surf.get_envelope(num_cyl=num_cyl, convex=False)
            ax.plot(env[:, 1], env[:, 0] * scale_envelope, c="g", linewidth=3)
        return ax

    def __call__(self, surface: Surface = Surface(), **kwargs):
        surface = super().__call__(surface=surface, **kwargs)
        surface = FourierSurfaceF(
            major_radius=self.get_major_radius(),
            file_path=self.file_path,
            **dict(surface),
        )
        return surface


class FourierSurfaceF(Surface):
    major_radius: ArrayLike
    num_tor_symmetry: int
    file_path: str

    def get_major_radius(self):
        return self.major_radius

    def get_minor_radius(self):
        return np.max(self.cartesian_to_toroidal()[:, :, 0])

    def cartesian_to_cylindrical(self):
        return cartesian_to_cylindrical(xyz=self.xyz)

    def cartesian_to_toroidal(self, xyz=None):
        if xyz is None:
            xyz = self.xyz
        return cartesian_to_toroidal(
            xyz=xyz,
            tore_radius=self.get_major_radius(),
        )

    def cartesian_to_shifted_cylindrical(self, xyz=None, num_cyl: int = 1, angle: float = 0.0):
        if xyz is None:
            xyz = self.xyz
        num_tor = xyz.shape[1]
        num_pt_cyl = num_tor // num_cyl
        rphiz_l = []
        for ind in range(num_cyl):
            xyz = xyz[:, (ind * num_pt_cyl) : ((ind + 1) * num_pt_cyl)]
            cyl_angle = -np.pi * (2 * ind + 1) / (self.num_tor_symmetry * num_cyl) + angle
            rphiz = cartesian_to_shifted_cylindrical(xyz=xyz, angle=cyl_angle, distance=self.get_major_radius())

            rphiz_l.append(rphiz)

        return onp.concatenate(rphiz_l, axis=1)

    def _get_rtheta(self, xyz=None, num_cyl: tp.Optional[int] = None, angle: float = 0.0):
        if xyz is None:
            xyz = self.xyz
        if num_cyl is None:
            rtheta = self.cartesian_to_toroidal(xyz=xyz)[..., :2]
        else:
            rtheta = self.cartesian_to_shifted_cylindrical(xyz=xyz, num_cyl=num_cyl, angle=angle)[..., :2]
        return rtheta

    def get_envelope(
        self,
        num_cyl: tp.Optional[int] = None,
        polar_coords: bool = True,
        convex: bool = True,
        angle: float = 0.0,
    ):
        rtheta = self._get_rtheta(num_cyl=num_cyl, angle=angle)

        points = np.reshape(rtheta, (-1, 2))
        xy_points = np.stack(from_polar(points[:, 0], points[:, 1])).T

        if convex:
            hull = ConvexHull(xy_points)
            sel_xy_points = xy_points[hull.vertices, :]
        else:
            sel_xy_points = concave_hull(xy_points, length_threshold=np.linalg.norm(xy_points[1] - xy_points[0]) * 4)
            sel_xy_points = np.stack(sel_xy_points)

        if not polar_coords:
            return sel_xy_points
        else:
            r, th = to_polar(sel_xy_points[:, 0], sel_xy_points[:, 1])
            rth = np.stack((r, th), axis=1)
            rth = rth[rth[:, 1].argsort()]
            return np.concatenate((rth, rth[:1]))

    def get_envelope_fourier_coeff(
        self,
        num_cyl: tp.Optional[int] = None,
        num_coeff: int = 5,
        convex: bool = False,
        angle: float = 0.0,
    ):
        xy = self.get_envelope(num_cyl=num_cyl, polar_coords=False, convex=convex, angle=angle)
        # import matplotlib.pyplot as plt;import seaborn as sns;import matplotlib;matplotlib.use('TkAgg')
        # plt.scatter(xy[:, 0], xy[:, 1]);plt.show()
        # import pdb;pdb.set_trace()
        r, th = to_polar(xy[:, 0], xy[:, 1])
        xy = xy[th.argsort()]
        th = th[th.argsort()]
        xy = np.concatenate((xy, xy[:1]))
        th = np.concatenate((th, th[:1] + 2 * np.pi))
        # rth_s = rth_s.at[-1, 1].set(rth_s[-1, 1] + 2 * np.pi)
        # interp = CubicSpline(th, xy, bc_type="periodic")
        interp = interp1d(th, xy, kind="linear", axis=0)
        interp = CubicSpline(th, xy, bc_type="periodic")

        def fun(theta):
            return to_polar(*interp(theta))[0]

        # fun = lambda x: onp.interp(x, rth_s[:, 1], rth_s[:, 0], period=2*np.pi)
        # import pdb;pdb.set_trace()
        # theta = np.linspace(0,2*np.pi, 50)
        # xy =interp(theta)
        # import matplotlib.pyplot as plt;import seaborn as sns;import matplotlib;matplotlib.use('TkAgg')
        # plt.scatter(xy[:, 0], xy[:, 1]);plt.show()

        return fourier_coefficients(th.min(), th.min() + 2 * np.pi, num_coeff, fun)

    def get_surface_envelope(
        self,
        num_cyl: tp.Optional[int] = None,
        num_coeff: int = 5,
        convex: bool = False,
        angle: float = 0.0,
    ):
        minor_radius, coefs = self.get_envelope_fourier_coeff(
            num_cyl=num_cyl, num_coeff=num_coeff, convex=convex, angle=angle
        )
        if num_cyl is None:
            return ToroidalSurface(
                integration_par=self.integration_par,
                num_tor_symmetry=self.num_tor_symmetry,
                major_radius=self.get_major_radius(),
                minor_radius=minor_radius,
                fourier_coeffs=coefs / minor_radius,
            )
        else:
            return CylindricalSurface(
                integration_par=self.integration_par,
                num_tor_symmetry=self.num_tor_symmetry * num_cyl,
                distance=self.get_major_radius(),
                radius=minor_radius,
                fourier_coeffs=coefs / minor_radius,
            )

    def get_gt_b_field(self, surface_labels: int = -1, b_norm_file: tp.Optional[str] = None):
        vmec = VMECIO.from_grid(
            self.file_path,
            ntheta=self.integration_par.num_points_u,
            nzeta=self.integration_par.num_points_v * self.num_tor_symmetry,
            surface_label=surface_labels,
        )
        if isinstance(surface_labels, int):
            b_field = vmec.b_cartesian[0]
        else:
            b_field = vmec.b_cartesian[surface_labels]

        b_field = b_field[:, : self.nbpts[1]]

        if b_norm_file is not None:
            bnorm = -vmec.scale_bnorm(get_bnorm(b_norm_file, self))
            b_field += bnorm[..., None] * self.normal_unit

        return b_field
