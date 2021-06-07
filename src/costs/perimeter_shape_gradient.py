import numpy as np
import configparser

from src.costs.abstract_shape_gradient import Abstract_shape_gradient
from src.costs.aux import f_e,grad_f_e

class Perimeter_shape_gradient(Abstract_shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config=config
        self.Np=int(config['geometry']['Np'])
        self.c0=float(config['optimization_parameters']['perim_c0'])
        self.c1=float(config['optimization_parameters']['perim_c1'])
    def cost(self,S):
        perim=self.Np*np.sum(S.dS)/S.npts
        return(f_e(self.c0,self.c1,perim))
    def shape_gradient(self,S,theta_peturbation):
        theta,dtildetheta,dtheta,dSdtheta=theta_peturbation
        perim=self.Np*np.sum(S.dS)/S.npts
        grad_perim=self.Np*np.einsum('oij->o',dSdtheta)/S.npts
        return grad_f_e(self.c0,self.c1,perim)*grad_perim