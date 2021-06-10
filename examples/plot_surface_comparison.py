import numpy as np
import configparser
import logging
from src.surface.surface_Fourier import Surface_Fourier
from src.costs.full_shape_gradient import Full_shape_gradient

logging.basicConfig(level=logging.INFO)
import pickle
def expand_for_plot(S):
    """from a toroidal_surface surface return X,Y,Z
    and add redundancy of first row"""
    shape=(S.X.shape[0]+1,S.X.shape[1])
    lst=[]
    for elt in [S.X,S.Y,S.Z]:
        new_elt=np.zeros(shape)
        new_elt[:-1,:]=elt
        new_elt[-1,:]=elt[0,:]
        lst.append(new_elt.copy())
    return lst


def plot(lst_S):
    from mayavi import mlab
    lst_s=[]
    for S in lst_S:
        X,Y,Z=expand_for_plot(S)
        lst_s.append(mlab.mesh(X,Y,Z,representation='mesh',colormap='Wistia'))
    mlab.show()

if __name__=='__main__':
    S_parametrization=Surface_Fourier.load_file('data/li383/cws.txt')
    path_config_file='config_file/config_full.ini'
    config = configparser.ConfigParser()
    config.read(path_config_file)
    config['other']['lamb']=str(5.1e-19)
    resolution=64
    S=Surface_Fourier(S_parametrization,(resolution,resolution),3)
    (m,n,Rmn,Zmn)=S_parametrization
    full_grad=Full_shape_gradient(config=config)
    full_grad.cost(np.concatenate((Rmn,Zmn)))
    
    with open('output/output_BFGS_incomplete','rb') as file:
        new_param=pickle.load(file)
    print(new_param.fun)
    full_grad.cost(new_param.x)
    (m,n,Rmn,Zmn)=S_parametrization
    R=new_param.x[:len(m)]
    Z=new_param.x[len(m):]
    S2=Surface_Fourier((m,n,R,Z),(resolution,resolution),3)
    
    Sp_parametrization=Surface_Fourier.load_file('data/li383/plasma_surf.txt')
    Sp=Surface_Fourier(Sp_parametrization,(resolution,resolution),3)
    plot([S2,Sp])