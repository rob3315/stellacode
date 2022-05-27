import numpy as np
from scipy.io import netcdf


def get_bnorm(path_bnorm, plasma):
    """get the normal field target from a file

    :param pathfile: path to extract
    :type pathfile: string
    :param plasma: the plasma
    :type plasma: Surface
    :rtype: 2D float array
    """
    if path_bnorm[-3::] == ".nc":
        f = netcdf.netcdf_file(path_bnorm, 'r', mmap=False)
        m = f.variables['xm_nyq'][()]
        n = f.variables['xn_nyq'][()] / plasma.Np
        bmn = f.variables['bmnc'][()][-1]
        f.close()
    else:
        data = []
        with open(path_bnorm, 'r') as f:
            for line in f:
                data.append(str.split(line))
        adata = np.array(data, dtype='float64')
        m, n, bmn = adata[:, 0], adata[:, 1], adata[:, 2]

    bnorm = np.zeros((plasma.grids[0]).shape)
    for i in range(len(m)):
        bnorm += bmn[i]*np.sin(2*np.pi*m[i]*plasma.grids[0] +
                               2*np.pi*n[i]*plasma.grids[1])
    return bnorm
