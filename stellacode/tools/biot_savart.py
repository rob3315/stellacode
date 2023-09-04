from typing import Optional

import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from stellacode import np
from .utils import eijk
import jax


@jax.jit
def biot_et_savart_op(
    xyz_plasma: ArrayLike,
    xyz_coil: ArrayLike,
    surface_current: ArrayLike,
    jac_xyz_coil: ArrayLike,
    dudv: float,
    plasma_normal: Optional[ArrayLike] = None,
) -> Array:
    """
    Args:
     * xyz_plasma: dims = up x vp x 3
     * xyz_coil: dims = uc x vc x 3
     * surface_current: dims = uc x vc x 3 or current_basis x uc x vc x 3
     * jac_xyz_coils: dims = uc x vc x 3 x 2
    """

    T = xyz_plasma[None, None, ...] - xyz_coil[:, :, None, None]
    K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

    # if plasma_normal is not None:
    #     B = np.einsum(
    #         "ijpqa,tijh,ijbh, dab,dpq->tpq",
    #         K,
    #         surface_current,
    #         jac_xyz_coil,
    #         eijk,
    #         plasma_normal,
    #     )
    # else:
    sc_jac = np.einsum("tijh,ijah->ijat", surface_current, jac_xyz_coil)
    B = np.einsum("ijpqa,ijbt,dba->tpqd", K, sc_jac, eijk)

    if plasma_normal is not None:
        B = np.einsum("tpqd,pqd->tpq", B, plasma_normal)
    return B * dudv


@jax.jit
def biot_et_savart(
    xyz_plasma: ArrayLike,
    xyz_coil: ArrayLike,
    j_3d: ArrayLike,
    dudv: float,
    plasma_normal: Optional[ArrayLike] = None,
) -> Array:
    """
    Args:
     * xyz_plasma: dims = up x vp x 3
     * xyz_coil: dims = uc x vc x 3
     * surface_current: dims = uc x vc x 3 or current_basis x uc x vc x 3
     * jac_xyz_coils: dims = uc x vc x 3 x 2
    """

    T = xyz_plasma[None, None, ...] - xyz_coil[:, :, None, None]
    K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

    B = np.einsum("ijpqa,ijb,dba->pqd", K, j_3d, eijk)
    if plasma_normal is not None:
        B = np.einsum("pqd,pqd->pq", B, plasma_normal)
    return B * dudv
