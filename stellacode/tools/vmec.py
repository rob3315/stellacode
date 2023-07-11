import typing as tp

from scipy.constants import mu_0
from scipy.io import netcdf_file

from stellacode import np


def _dot(coeff, cossin):
    return np.einsum("rm,tzm->tzr", coeff, cossin)


class VMECIO:
    def __init__(self, vmec_file: str):
        self.file = netcdf_file(vmec_file)

    def get_val(
        self,
        key: str,
        theta: np.array,
        zeta: np.array,
        surf_labels: tp.Optional[tp.List[int]] = None,
        diff: tp.Optional[str] = None,
    ):
        xm = self.file.variables["xm_nyq"][()].astype(int)

        xn = self.file.variables["xn_nyq"][()].astype(int)
        angle = theta[..., None] * xm - zeta[..., None] * xn

        val = 0.0
        if f"{key}mnc" in self.file.variables.keys():
            mnc = self.file.variables[f"{key}mnc"][()].astype(float)
            if surf_labels is not None:
                mnc = mnc[surf_labels]
            if diff == "theta":
                val -= _dot(mnc * xm[:, None], np.sin(angle))
            elif diff == "zeta":
                val += _dot(mnc * xn[:, None], np.sin(angle))
            else:
                val += _dot(mnc, np.cos(angle))

        if f"{key}mns" in self.file.variables.keys():
            mns = self.file.variables[f"{key}mns"][()].astype(float)
            if surf_labels is not None:
                mns = mns[surf_labels]
            if diff == "theta":
                val += _dot(mns * xm[:, None], np.cos(angle))
            elif diff == "zeta":
                val -= _dot(mns * xn[:, None], np.cos(angle))
            else:
                val += _dot(mns, np.sin(angle))

        return val

    def get_val_grad(
        self,
        key: str,
        theta: np.array,
        zeta: np.array,
        surf_labels: tp.Optional[tp.List[int]] = None,
    ):
        kwargs = dict(key=key, theta=theta, zeta=zeta, surf_labels=surf_labels)
        return np.stack(
            (
                self.get_val(**kwargs, diff="theta"),
                self.get_val(**kwargs, diff="zeta"),
            ),
            axis=-1,
        )

    def get_b_vmec(
        self,
        theta: np.array,
        zeta: np.array,
        surf_labels: tp.Optional[tp.List[int]] = None,
        covariant: bool = True,
    ):
        kwargs = dict(theta=theta, zeta=zeta, surf_labels=surf_labels)
        if covariant:
            prefix = "bsub"
        else:
            prefix = "bsup"
        return np.stack(
            (
                self.get_val(f"{prefix}s", **kwargs),
                self.get_val(f"{prefix}u", **kwargs),
                self.get_val(f"{prefix}v", **kwargs),
            ),
            axis=-1,
        )

    def get_b_cylindrical(
        self,
        theta: np.array,
        zeta: np.array,
        surf_labels: tp.Optional[tp.List[int]] = None,
    ):
        kwargs = dict(theta=theta, zeta=zeta, surf_labels=surf_labels)
        r_grad = self.get_val_grad("r", **kwargs)
        z_grad = self.get_val_grad("z", **kwargs)
        b_cov = self.get_b_vmec(**kwargs, covariant=True)
        radius = self.get_val("r", **kwargs)

        return np.stack(
            (
                np.einsum("tzmc,tzmc->tzm", b_cov[..., :2], r_grad),
                radius * b_cov[..., 1],
                np.einsum("tzmc,tzmc->tzm", b_cov[..., :2], z_grad),
            ),
            axis=-1,
        )

    def get_b_cartesian(
        self,
        theta: np.array,
        zeta: np.array,
        surf_labels: tp.Optional[tp.List[int]] = None,
    ):
        b_cyl = self.get_b_cylindrical(theta=theta, zeta=zeta, surf_labels=surf_labels)
        zeta_ = zeta[..., None]

        return np.stack(
            (
                b_cyl[..., 0] * np.cos(zeta_) - b_cyl[..., 1] * np.sin(zeta_),
                b_cyl[..., 0] * np.sin(zeta_) + b_cyl[..., 1] * np.cos(zeta_),
                b_cyl[..., -1],
            ),
            axis=-1,
        )

    def get_nfp(self):
        return self.file.variables["nfp"][()].astype(int)

    def scale_bnorm(self, b_norm):
        """
        From regcoil:
        BNORM scales B_n by curpol=(2*pi/nfp)*bsubv(m=0,n=0)
        where bsubv is the extrapolation to the last full mesh point of
        bsubvmnc.  Let's undo this scaling now.
        """
        nfp = self.get_nfp()
        bsubvmnc = self.file.variables["bsubvmnc"][()].astype(float)

        bsubv00 = 1.5 * bsubvmnc[0, -1] - 0.5 * bsubvmnc[0, -2]
        curpol = 2 * np.pi / nfp * bsubv00
        return b_norm * curpol

    def get_net_poloidal_current(self):
        """
        From regcoil:
        bvco seems related to the covariant components of B bsubvmnc/s
        In principle this value should be given by:
        2pi/mu_0*integral_zeta(B(theta=interior, radius=last_radius))
        """

        bvco = self.file.variables["bvco"][()].astype(float)
        return 2 * np.pi / mu_0 * (1.5 * bvco[-1] - 0.5 * bvco[-2])
