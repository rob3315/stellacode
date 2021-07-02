import numpy as np
def get_bnorm(pathfile,plasma):
    """get the normal field target from a file

    :param pathfile: path to extract
    :type pathfile: string
    :param plasma: the plasma
    :type plasma: Surface
    :rtype: 2D float array
    """
    data=[]
    with open(pathfile,'r') as f:
        for line in f:
            data.append(str.split(line))
    adata=np.array(data,dtype='float64')
    m,n,bmn=adata[:,0],adata[:,1],adata[:,2]
    bnorm=np.zeros((plasma.grids[0]).shape)
    for i in range(len(m)):
        bnorm+=bmn[i]*np.sin(2*np.pi*m[i]*plasma.grids[0]+2*np.pi*n[i]*plasma.grids[1])
    return bnorm
