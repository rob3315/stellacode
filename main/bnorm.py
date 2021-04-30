import numpy as np
def get_bnorm(pathfile,plasma):
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
if __name__=='__main__':
    from toroidal_surface import *
    P_parametrization=Toroidal_surface.load_file('code/data/li383/cws.txt')
    plasma=Toroidal_surface(P_parametrization,(128,128),3)
    pathfile='code/data/li383/bnorm.txt'
    b=get_bnorm(pathfile,plasma)