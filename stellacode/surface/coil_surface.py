from pydantic import BaseModel, Extra, Field
import typing as tp
from stellacode import np
from jax.typing import ArrayLike
from .abstract_surface import AbstractSurface
from .current import AbstractCurrent
from stellacode.tools.utils import get_min_dist


class CoilSurface(BaseModel):
    """A class used to:
    * represent an abstract surfaces
    * computate of the magnetic field generated by a current carried by the surface.
    * visualize surfaces
    """

    surface: AbstractSurface
    current: AbstractCurrent
    current_op: tp.Optional[ArrayLike] = None
    xyz: tp.Optional[ArrayLike] = None
    jac_xyz: tp.Optional[ArrayLike] = None
    normal: tp.Optional[ArrayLike] = None
    normal_unit: tp.Optional[ArrayLike] = None
    ds: tp.Optional[ArrayLike] = None
    principles: tp.Optional[tp.Tuple[ArrayLike, ArrayLike]] = None
    # mult_current_grid: tp.Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow  # allow extra fields

    def __init__(self, **kwargs):
        # if self.mult_current_grid is not None:
        #     self.surface.nbpts = (
        #         self.mult_current_grid * self.current.num_pol,
        #         self.mult_current_grid * self.current.num_tor,
        #     )
        super().__init__(**kwargs)
        self.compute_surface_attributes()
        # self.current_op = self.current.get_matrix_from_grid(self.grids)

    @classmethod
    def from_config(cls, config):
        from .imports import get_current_potential, get_cws_grid

        current = get_current_potential(config)
        surface = get_cws_grid(config)

        return cls(surface=surface, current=current)

    def compute_surface_attributes(self, deg=2):
        raise NotImplementedError

    def get_curent_op(self):
        return self.current.get_matrix_from_grid(self.grids)

    def get_current_scalar_prod(self):
        return compute_Qj(self.current_op, self.jac_xyz, self.ds)

    @property
    def nbpts(self):
        return self.surface.nbpts

    @property
    def npts(self):
        return self.surface.npts
        # return self.nbpts[0] * self.nbpts[1]

    @property
    def dudv(self):
        return self.surface.dudv

    def get_distance(self, xyz):
        return np.linalg.norm(self.xyz[..., None, None, :] - xyz[None, None, ...], axis=-1)

    def get_min_distance(self, xyz):
        return get_min_dist(self.xyz, xyz)
    
    def get_j_3D(self, phi_mn):
        # phi_mn is a vector containing the components of the best scalar current potential.
        # The real surface current is given by :
        return np.einsum("oijk,ijdk,ij,o->ijd", self.current_op, self.jac_xyz, 1 / self.ds, phi_mn)

    def get_j_surface(self, phi_mn):
        return np.einsum("oijk,ij,o->ijk", self.current_op, 1 / self.ds, phi_mn)

    def plot_j_surface(self, phi_mn, num_rot: int = 3):
        import matplotlib.pyplot as plt

        j_surface = self.get_j_surface(phi_mn)
        j_norm = np.linalg.norm(j_surface, axis=-1)
        num_rot = min(num_rot, j_norm.shape[1] // self.grids[0].shape[1])
        shape_block = self.grids[0].shape[1]

        for i in range(num_rot):
            ax = plt.subplots()[1]
            id1 = shape_block * i
            id2 = shape_block * (i + 1)
            ax.quiver(
                self.grids[1] * 2 * np.pi,
                self.grids[0] * 2 * np.pi,
                j_surface[:, id1:id2, 1],
                j_surface[:, id1:id2, 0],
                j_norm[:, id1:id2],
                units="width",
            )
            plt.xlabel("Poloidal angle")
            plt.ylabel("Toroidal angle")


def compute_Qj(matrixd_phi, dpsi, dS):
    """take only the segment whitout rotation of j"""
    lu, lv = dS.shape
    Qj = np.einsum(
        "oija,ijda,ijdk,pijk,ij->op",
        matrixd_phi,
        dpsi,
        dpsi,
        matrixd_phi,
        1 / dS,
        optimize=True,
    ) / (lu * lv)
    return Qj
