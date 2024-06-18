import jax.numpy as np
import numpy as onp


from stellacode import np


def get_eijk():
    """
    Returns the completely antisymmetric tensor.

    Returns
    -------
    eijk : numpy.ndarray
        The completely antisymmetric tensor of shape (3, 3, 3).
    """
    # Initialize the tensor with zeros
    eijk = onp.zeros((3, 3, 3))

    # Set the elements of the tensor
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2,
                                         0, 1] = 1  # Upper diagonal elements
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = - \
        1  # Lower diagonal elements

    # Convert to jax.numpy array
    eijk = np.asarray(eijk)

    return eijk


eijk = get_eijk()


def cfourier1D(nzeta, fmnc, xn):
    cfunct = []
    for i in range(nzeta):
        ci = 0
        zeta = i*2*np.pi/nzeta
        for x, mode in zip(xn, fmnc):
            angle = - x*zeta
            ci += mode*np.cos(angle)
        cfunct.append(ci)
    return np.array(cfunct)


def sfourier1D(nzeta, fmns, xn):
    sfunct = []
    for i in range(nzeta):
        si = 0
        zeta = i*2*np.pi/nzeta
        for x, mode in zip(xn, fmns):
            angle = - x*zeta
            si += mode*np.sin(angle)
        sfunct.append(si)
    return np.array(sfunct)
