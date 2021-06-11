import logging
import numpy as np
import configparser
from src.surface.surface_Fourier import Surface_Fourier
from src.costs.abstract_shape_gradient import Abstract_shape_gradient
from src.costs.aux import f_e,grad_f_e

class Curvature_shape_gradient(Abstract_shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config=config
        self.Np=int(config['geometry']['Np'])
        self.ntheta_coil   = int(config['geometry']['ntheta_coil'])
        self.nzeta_coil   = int(config['geometry']['nzeta_coil'])
        self.c0=float(config['optimization_parameters']['curvature_c0'])
        self.c1=float(config['optimization_parameters']['curvature_c1'])
        self.vf = np.vectorize(lambda x : f_e(self.c0,self.c1,x))
        
    def cost(self, S):
        abs_curv=np.maximum(np.abs(S.principles[0]),np.abs(S.principles[1]))
        cost=self.Np*np.einsum('ij,ij->',self.vf(abs_curv),S.dS/S.npts)
        logging.info('maximal curvature {:5e} m^-1, curvature cost : {:5e}'.format(np.max(abs_curv),cost))
        return cost
    def curvature_derivative(self,S,theta_peturbation):
        dtheta=theta_peturbation['dtheta']
        result={}
        dE=2*np.einsum('lij,oijl->oij', S.dpsi[0], dtheta[:,:,:,0,:]) 
        dF=np.einsum('lij,oijl->oij', S.dpsi[0], dtheta[:,:,:,1,:])+np.einsum('lij,oijl->oij', S.dpsi[1], dtheta[:,:,:,0,:]) 
        dG=2*np.einsum('lij,oijl->oij', S.dpsi[1], dtheta[:,:,:,1,:]) 
        (E,F,G)=S.I
        result['dI']=(dE,dF,dG)
        return result

    def shape_gradient(self, S, theta_pertubation):
        pass