import numpy as onp


def get_bnorm(path_bnorm: str, plasma):
    """
    Compute the target magnetic field from a file generated by BNORM code from STELLOPT.
    The target is the normal inward magnetic field on the plasma surface.
    This magnetic field is normalized by curpol : the real target magnetic field is curpol * bnorm.
    The curpol number is found in the nescin file generated by BNORM.

    Args:
        * path_bnorm: path to bnorm file
        * plasma: the plasma surface

    Returns:
        * normal B field on the plasma surface

    """
    data = []
    with open(path_bnorm, "r") as f:
        for line in f:
            data.append(str.split(line))
    adata = onp.array(data, dtype="float64")
    m, n, bmn = adata[:, 0], adata[:, 1], adata[:, 2]

    bnorm = onp.zeros((plasma.grids[0]).shape)
    for i in range(len(m)):
        bnorm += bmn[i] * onp.sin(2 * onp.pi * m[i] * plasma.grids[0] + 2 * onp.pi * n[i] * plasma.grids[1])
    return bnorm