import numpy as np
import configparser

from src.costs.abstract_shape_gradient import Abstract_shape_gradient
from src.costs.aux import f_non_linear,grad_f_non_linear
import src.tools as tools

class Distance_shape_gradient(Abstract_shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config=config
        self.Np=int(config['geometry']['Np'])
        self.d_min_hard = float(config['optimization_parameters']['d_min_hard'])
        self.d_min_soft= float(config['optimization_parameters']['d_min_soft'])
        self.d_min_penalization= float(config['optimization_parameters']['d_min_penalization'])
        self.vf = np.vectorize(lambda x : f_non_linear(self.d_min_hard,self.d_min_soft,self.d_min_penalization,x))
        self.rot_tensor=tools.get_rot_tensor(self.Np)
    def cost(self, S, T):
        dist=np.linalg.norm(T,axis=-1)
        dist_min=np.amin(dist,axis=(0,3,4))
        return self.Np*np.einsum('ij,ij->',self.vf(dist_min),S.dS/S.npts)
    def shape_gradient(self, S, T, theta_pertubation):
        #compute the necessary tools for the gradient of the cost distance,
        # the gradient is \int_S X dtheta dS + \int_S Y dS/dtheta
        ls,lu,lv,lu_plasma,lv_plasma,_=T.shape
        dist=np.linalg.norm(T,axis=-1)
        dist_min=np.min(dist,axis=(0,3,4))
        dist_=np.einsum('sijpq->ijspq',dist)
        indexes=np.argmin(np.reshape(dist_,(lu,lv,-1)),axis=-1)
        aux= lambda  x : np.unravel_index(x,(ls,lu_plasma,lv_plasma))
        v_aux=np.vectorize(aux)
        indexes_fulls,indexes_fullp,indexes_fullq=v_aux(indexes)
        T_min=np.zeros((lu,lv,3))
        vgradf=np.zeros((lu,lv))
        Y=np.zeros((lu,lv))
        for i in range(lu):
            for j in range(lv):
                T_min[i,j,:]=self.rot_tensor[(ls-indexes_fulls[i,j])%ls]@ T[indexes_fulls[i,j],i,j,indexes_fullp[i,j],indexes_fullq[i,j],:]
                vgradf[i,j]=grad_f_non_linear(self.d_min_hard,self.d_min_soft,self.d_min_penalization,dist_min[i,j])
                Y[i,j]=f_non_linear(self.d_min_hard,self.d_min_soft,self.d_min_penalization,dist_min[i,j])
        grad_d_min=T_min/dist_min[:,:,np.newaxis]
        X=-1*vgradf[:,:,np.newaxis]*grad_d_min

        theta,dtildetheta,dtheta,dSdtheta=theta_pertubation
        grad_distance=self.Np*(np.einsum('oijl,ijl,ij->o',theta,X,S.dS/S.npts)+np.einsum('ij,oij->o',Y,dSdtheta/S.npts))
        return grad_distance
        
