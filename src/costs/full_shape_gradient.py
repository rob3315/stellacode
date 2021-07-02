import logging
import numpy as np
import configparser

import src.costs.EM_shape_gradient as EM
from src.costs.distance_shape_gradient import Distance_shape_gradient
from src.costs.perimeter_shape_gradient import Perimeter_shape_gradient
from src.costs.curvature_shape_gradient import Curvature_shape_gradient
from src.surface.surface_Fourier import Surface_Fourier
# The main object of 
class Full_shape_gradient():
    """Put together all the costs"""
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config=config
        self.Np=int(config['geometry']['Np'])
        self.ntheta_coil   = int(config['geometry']['ntheta_coil'])
        self.nzeta_coil   = int(config['geometry']['nzeta_coil'])

        # Initialization of the different costs :
        self.EM=EM.EM_shape_gradient(config=config)
        (m,n,Rmn,Zmn)=self.EM.S_parametrization
        self.m,self.n=m,n
        self.init_param=np.concatenate((Rmn,Zmn))
        self.lst_cost=[self.EM]
        if config['optimization_parameters']['d_min']=='True':
            self.dist=Distance_shape_gradient(config=config)
            self.lst_cost.append(self.dist)
        if config['optimization_parameters']['perim']=='True':
            self.perim=Perimeter_shape_gradient(config=config)
            self.lst_cost.append(self.perim)
        if config['optimization_parameters']['curvature']=='True':
            self.curv=Curvature_shape_gradient(config=config)
            self.lst_cost.append(self.curv)
        
    def cost(self,param_S_array):
        R=param_S_array[:len(self.m)]
        Z=param_S_array[len(self.m):]
        paramS=((self.m,self.n,R,Z))
        S=Surface_Fourier(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        c=0
        for elt in self.lst_cost:
            new_cost,_=elt.cost(S)
            c+=new_cost
        logging.info('Total cost : {:5e}'.format(c))
        return c
    def shape_gradient(self,param_S_array):
        """Full_shape_gradient only needs the surface parametrization

        :param param_S_array:
        :type param_S_array: 1D array
        :return: the shape gradient
        :rtype: 1D array
        """
        R=param_S_array[:len(self.m)]
        Z=param_S_array[len(self.m):]
        paramS=((self.m,self.n,R,Z))
        S=Surface_Fourier(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        theta_pertubation=S.get_theta_pertubation()
        shape_grad=(self.lst_cost[0]).shape_gradient(S,theta_pertubation)
        for elt in self.lst_cost[1:]:
            shape_grad+=elt.shape_gradient(S,theta_pertubation)
        return shape_grad
