import numpy as np
import configparser
import logging
from src.surface.surface_Fourier import Surface_Fourier
from src.costs.full_shape_gradient import Full_shape_gradient
from src.surface.surface_Fourier import plot

logging.basicConfig(level=logging.INFO)
import pickle


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