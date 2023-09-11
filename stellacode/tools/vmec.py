import typing as tp

import numpy as np
from pydantic import BaseModel
from scipy.constants import mu_0
from scipy.io import netcdf_file

from .coordinates import cylindrical_to_cartesian, vmec_to_cylindrical

# from functools import cache


def _dot(coeff, cossin):
    return np.einsum("rm,tzm->rtz", coeff, cossin)


class Grid(BaseModel):
    theta: np.ndarray
    zeta: np.ndarray
    surf_labels: tp.Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False  # required for caching
        frozen = True


class VMECIO:
    def __init__(self, vmec_file: str, grid: Grid):
        self.file = netcdf_file(vmec_file)
        self.grid = grid

    def get_var(self, key: str, cast_type=float):
        return self.file.variables[key][()].astype(cast_type)

    # @cache
    def get_val(
        self,
        key: str,
        diff: tp.Optional[str] = None,
        nyq: bool = True,
    ):
        if nyq:
            suffix = "_nyq"
        else:
            suffix = ""
        xm = self.get_var(f"xm{suffix}", int)
        xn = self.get_var(f"xn{suffix}", int)

        angle = self.grid.theta[..., None] * xm - self.grid.zeta[..., None] * xn

        val = 0.0
        if f"{key}mnc" in self.file.variables.keys():
            mnc = self.get_var(f"{key}mnc", float)
            if self.grid.surf_labels is not None:
                mnc = mnc[self.grid.surf_labels]
            if diff == "theta":
                val -= _dot(mnc * xm[None, :], np.sin(angle))
            elif diff == "zeta":
                val += _dot(mnc * xn[None, :], np.sin(angle))
            else:
                val += _dot(mnc, np.cos(angle))

        if f"{key}mns" in self.file.variables.keys():
            mns = self.get_var(f"{key}mns", float)
            if self.grid.surf_labels is not None:
                mns = mns[self.grid.surf_labels]
            if diff == "theta":
                val += _dot(mns * xm[None, :], np.cos(angle))
            elif diff == "zeta":
                val -= _dot(mns * xn[None, :], np.cos(angle))
            else:
                val += _dot(mns, np.sin(angle))

        return val

    def get_val_grad(
        self,
        key: str,
        nyq: bool = True,
    ):
        kwargs = dict(key=key, nyq=nyq)
        return np.stack(
            (
                self.get_val(**kwargs, diff="theta"),
                self.get_val(**kwargs, diff="zeta"),
            ),
            axis=-1,
        )

    def get_b_vmec(
        self,
        covariant: bool = True,
    ):
        if covariant:
            prefix = "bsub"
            bs = self.get_val(f"bsubs")
        else:
            prefix = "bsup"
            lt, lz = self.grid.theta.shape
            bs = np.zeros((self.grid.surf_labels.shape[0], lt, lz))
        return np.stack(
            (self.get_val(f"{prefix}u"), self.get_val(f"{prefix}v"), bs),
            axis=-1,
        )

    @property
    def rphiz(self):
        radius = self.get_val("r", nyq=False)
        z_ = self.get_val("z", nyq=False)
        phi = np.tile(self.grid.zeta, (radius.shape[0], 1, 1))
        return np.stack((radius, phi, z_), axis=-1)

    @property
    def xyz(self):
        rphiz = self.rphiz
        return cylindrical_to_cartesian(rphiz, self.grid.zeta)

    @property
    def grad_rphiz(self):
        r_grad = self.get_val_grad("r", nyq=False)
        z_grad = self.get_val_grad("z", nyq=False)
        grad_rpz = np.stack((r_grad, np.zeros_like(r_grad), z_grad), axis=-1)
        return np.transpose(grad_rpz, (0, 1, 2, 4, 3))

    @property
    def grad_xyz(self):
        grad_rphiz = self.grad_rphiz
        rphiz = self.rphiz[..., None]
        gr, gz = grad_rphiz[..., 0, :], grad_rphiz[..., 2, :]
        r_, phi = rphiz[..., 0, :], rphiz[..., 1, :]

        return np.stack(
            (
                gr * np.cos(phi) - r_ * np.sin(phi),
                gr * np.sin(phi) + r_ * np.cos(phi),
                gz,
            ),
            axis=-2,
        )

    def vmec_to_cylindrical(self, vector_field):
        grad_rphiz = self.grad_rphiz
        rphiz = self.rphiz
        return vmec_to_cylindrical(vector_field, rphiz, grad_rphiz)

    @property
    def b_cylindrical(
        self,
    ):
        return self.vmec_to_cylindrical(self.get_b_vmec(covariant=False))

    @property
    def b_cartesian(
        self,
    ):
        return cylindrical_to_cartesian(self.b_cylindrical, self.grid.zeta)

    @property
    def j_vmec(
        self,
    ):
        ju = self.get_val(f"curru")
        jv = self.get_val(f"currv")
        return np.stack((ju, jv, np.zeros_like(ju)), axis=-1)

    @property
    def j_cylindrical(
        self,
    ):
        return self.vmec_to_cylindrical(self.j_vmec)

    @property
    def j_cartesian(
        self,
    ):
        return cylindrical_to_cartesian(self.j_cylindrical, self.grid.zeta)

    @property
    def nfp(self):
        return self.get_var("nfp", int)

    @property
    def curpol(self):
        bsubvmnc = self.get_var("bsubvmnc", float).T

        bsubv00 = 1.5 * bsubvmnc[0, -1] - 0.5 * bsubvmnc[0, -2]
        curpol = 2 * np.pi / self.nfp * bsubv00
        return curpol

    def scale_bnorm(self, b_norm):
        """
        From regcoil:
        BNORM scales B_n by curpol=(2*pi/nfp)*bsubv(m=0,n=0)
        where bsubv is the extrapolation to the last full mesh point of
        bsubvmnc.  Let's undo this scaling now.
        """
        return b_norm * self.curpol

    @property
    def net_poloidal_current(self):
        """
        From regcoil:
        bvco seems related to the covariant components of B bsubvmnc/s
        In principle this value should be given by:
        2pi/mu_0*integral_zeta(B(theta=interior, radius=last_radius))
        """

        bvco = self.get_var("bvco", float)
        return 2 * np.pi / mu_0 * (1.5 * bvco[-1] - 0.5 * bvco[-2])

    @property
    def net_poloidal_current2(self):
        xyz_dv = self.grad_xyz[-1, :, :, :, 1] * (2 * np.pi)
        return np.mean(np.sum((self.b_cartesian[-1] * xyz_dv)[:, 0], axis=-1), 0) / mu_0

    @classmethod
    def from_grid(
        cls,
        file_path,
        ntheta: int = 10,
        nzeta: int = 10,
        n_surf_label: tp.Optional[int] = None,
        surface_label: tp.Optional[int] = None,
    ):
        file_ = netcdf_file(file_path)
        theta_ = np.linspace(0, 2 * np.pi, num=ntheta)
        zeta_ = np.linspace(0, 2 * np.pi, num=nzeta)
        zeta, theta = np.meshgrid(zeta_, theta_)
        ns = file_.variables["ns"][()].astype(int)
        if surface_label is None:
            if n_surf_label is None:
                n_surf_label = ns
            surf_labels = np.linspace(0, ns - 1, num=n_surf_label, endpoint=True).round().astype(int)
        else:
            surf_labels = np.array([surface_label])
        return cls(file_path, grid=Grid(theta=theta, zeta=zeta, surf_labels=surf_labels))
