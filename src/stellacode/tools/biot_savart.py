from typing import Optional

import jax
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from stellacode import np, mu_0_fac

from .utils import eijk


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
    Compute the magnetic field vector on the surface defined by xyz_plasma
    caused by surface_current on the surface defined by xyz_coil.

    Args:
     * xyz_plasma: dims = up x vp x 3
     * xyz_coil: dims = uc x vc x 3
     * surface_current: dims = uc x vc x 3 or current_basis x uc x vc x 3
     * jac_xyz_coil: dims = uc x vc x 3 x 2
     * dudv: scaling factor for the magnetic field
     * plasma_normal: plasma normal vector used to project the magnetic field on the
        normal component (optional).

    Returns:
        A 3D array with dimensions up x vp x 3 with the magnetic field on the
        plasma surface caused by the surface current on the coil surface.
    """

    # Compute the vector T from the plasma surface to the coil surface
    T = xyz_plasma[None, None, ...] - xyz_coil[:, :, None, None]

    # Precompute the scalar K = T / ||T||^3
    K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

    # Compute the surface current on the coil surface in the basis of the
    #  Jacobian matrix of the coil surface
    sc_jac = np.einsum("tijh,ijah->ijat", surface_current, jac_xyz_coil)

    # Compute the magnetic field vector on the plasma surface
    # B = \sum_i \int \frac{J_i}{||R - b_i||^3} \times (R - b_i) db_i
    # where J_i is the surface current, b_i is a point on the coil surface
    # and R is a point on the plasma surface
    B = np.einsum("ijpqa,ijbt,dba->tpqd", K, sc_jac, eijk)

    # If a plasma normal vector is provided, project the magnetic field on the
    # normal component
    if plasma_normal is not None:
        B = np.einsum("tpqd,pqd->tpq", B, plasma_normal)

    # Scale the magnetic field by the scaling factor dudv
    return B * dudv


@jax.jit
def biot_et_savart(
    xyz_plasma: ArrayLike,
    xyz_coil: ArrayLike,
    j_3d: ArrayLike,
    dudv: float,
    plasma_normal: Optional[ArrayLike] = None
) -> Array:
    """
    Compute the magnetic field on the plasma surface caused by the surface current on the coil surface.

    Args:
     * xyz_plasma: dims = up x vp x 3
     * xyz_coil: dims = uc x vc x 3
     * j_3d: dims = uc x vc x 3. Surface current on the coil surface.
     * dudv: scaling factor for the magnetic field
     * plasma_normal: plasma normal vector used to project the magnetic field on the
        normal component (optional).

    Returns:
        A 3D array with dimensions up x vp x 3 with the magnetic field on the
        plasma surface caused by the surface current on the coil surface.
    """

    # Compute the vector T from the plasma surface to the coil surface
    T = xyz_plasma[None, None, ...] - xyz_coil[:, :, None, None]

    # Precompute the scalar K = T / ||T||^3
    K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

    # Compute the magnetic field vector on the plasma surface
    # B = \sum_i \int \frac{J_i}{||R - b_i||^3} \times (R - b_i) db_i
    # where J_i is the surface current, b_i is a point on the coil surface
    # and R is a point on the plasma surface
    B = np.einsum("ijpqa,ijb,dba->pqd", K, j_3d, eijk)

    # If a plasma normal vector is provided, project the magnetic field on the
    # normal component
    if plasma_normal is not None:
        B = np.einsum("pqd,pqd->pq", B, plasma_normal)

    # Scale the magnetic field by the scaling factor dudv
    return B * dudv
