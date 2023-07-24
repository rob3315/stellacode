import typing as tp
from os import sep

import numpy as onp
from jax.typing import ArrayLike
from scipy.interpolate import CubicSpline, interp1d
from scipy.io import netcdf_file
from scipy.spatial import ConvexHull

from stellacode import np
from stellacode.surface.utils import fourier_coefficients
from concave_hull import concave_hull
from .abstract_surface import AbstractSurface, IntegrationParams
from .tore import ToroidalSurface
from .utils import cartesian_to_cylindrical, cartesian_to_toroidal, from_polar, to_polar
from stellacode.tools.vmec import VMECIO


class FourierSurface(AbstractSurface):
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
    def from_file(cls, path_surf, integration_par, n_fp=1):
        """This function returns a Surface_Fourier object defined by a file.
        Three kinds of file are currently supported :
        - wout_.nc files generated by VMEC (for plasma surfaces)
        - nescin files generated by regcoil
        - text files (see example in data/li383)

        load file with the format m,n,Rmn,Zmn"""

        if path_surf[-3::] == ".nc":
            vmec = VMECIO.from_grid(path_surf)
            m = vmec.get_var("xm", int)
            n = -vmec.get_var("xn", int) / vmec.nfp
            Rmn = vmec.get_var("rmnc", float)[-1]
            Zmn = vmec.get_var("zmns", float)[-1]
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

    def get_minor_radius(self):
        return np.max(self.cartesian_to_toroidal()[:, :, 0])

    def cartesian_to_cylindrical(self):
        return cartesian_to_cylindrical(xyz=self.xyz)

    def cartesian_to_toroidal(self):
        return cartesian_to_toroidal(
            xyz=self.xyz,
            tore_radius=self.get_major_radius(),
            height=self.Zmn[0],
        )

    def get_axisymmetric_envelope(self, polar_coords: bool = True, convex: bool = True):
        # Could also find the concave hull as shown here:
        # https://stackoverflow.com/questions/57260352/python-concave-hull-polygon-of-a-set-of-lines

        rtphi = self.cartesian_to_toroidal()

        points = np.reshape(rtphi[..., :2], (-1, 2))
        xy_points = np.stack(from_polar(points[:, 0], points[:, 1])).T
        if convex:
            hull = ConvexHull(xy_points)
            sel_xy_points = xy_points[hull.vertices, :]
        else:
            sel_xy_points = concave_hull(xy_points, length_threshold=0.2)
            sel_xy_points = np.stack(sel_xy_points)

        if not polar_coords:
            return sel_xy_points
        else:
            r, th = to_polar(sel_xy_points[:, 0], sel_xy_points[:, 1])
            rth = np.stack((r, th), axis=1)
            rth = rth[rth[:, 1].argsort()]
            return np.concatenate((rth, rth[:1]))

    def get_axisymmetric_evelope_fourier_coeff(self, num_coeff: int = 5, convex: bool = False):
        xy = self.get_axisymmetric_envelope(polar_coords=False, convex=convex)
        r, th = to_polar(xy[:, 0], xy[:, 1])
        xy = xy[th.argsort()]
        th = th[th.argsort()]
        xy = np.concatenate((xy, xy[:1]))
        th = np.concatenate((th, th[:1] + 2 * np.pi))
        # rth_s = rth_s.at[-1, 1].set(rth_s[-1, 1] + 2 * np.pi)
        # interp = CubicSpline(th, xy, bc_type="periodic")
        interp = interp1d(th, xy, kind="linear", axis=0)

        def fun(theta):
            return to_polar(*interp(theta))[0]

        # fun = CubicSpline(rth_s[:, 1], rth_s[:, 0], bc_type="periodic")
        # fun = lambda x: onp.interp(x, rth_s[:, 1], rth_s[:, 0], period=2*np.pi)

        return fourier_coefficients(th.min(), th.min() + 2 * np.pi, num_coeff, fun)

    def get_axisymmetric_surface_envelope(self, num_coeff: int = 5, convex: bool = False):
        minor_radius, coefs = self.get_axisymmetric_evelope_fourier_coeff(num_coeff=num_coeff, convex=convex)
        return ToroidalSurface(
            integration_par=self.integration_par,
            num_tor_symmetry=self.num_tor_symmetry,
            major_radius=self.get_major_radius(),
            minor_radius=minor_radius,
            fourier_coeffs=coefs / minor_radius,
        )

    def plot_cross_sections(self, num: int = 5, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        rtphi = self.cartesian_to_toroidal()

        num_phi_s = rtphi.shape[1]
        for i in range(0, num_phi_s, num_phi_s // num):
            rphi = np.concatenate((rtphi[:, i, :], rtphi[:1, i, :]), axis=0)
            ax.plot(rphi[:, 1], rphi[:, 0], c=[0, 0] + [i / num_phi_s])

        env = self.get_axisymmetric_envelope(convex=True)
        ax.plot(env[:, 1], env[:, 0], c="r", linewidth=3)
        env = self.get_axisymmetric_envelope(convex=False)
        ax.plot(env[:, 1], env[:, 0], c="g", linewidth=3)
        return ax
