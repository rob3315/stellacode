import numpy as np
import configparser
from src.costs import EM_shape_gradient
from toroidal_surface import *
import cost_surface
import tools
# The main object of 
class Shape_gradient(EM_shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        super().__init__(self,config=config)
        # Initialization of the different costs :
        
        self.d_min_hard = float(config['optimization_parameters']['d_min_hard'])
        self.d_min_soft= float(config['optimization_parameters']['d_min_soft'])
        self.d_min_penalization= float(config['optimization_parameters']['d_min_penalization'])
    def distance_cost(self,T,S):
        dist=np.linalg.norm(T,axis=-1)
        return Full_gradient.non_linear_cost(self.d_min_hard,self.d_min_soft,self.d_min_penalization,dist,S.dS)
    def gradient_cost_distance(self,T,S):
        #compute the necessary tools for the gradient of the cost distance,
        # the gradient is \int_S X dtheta dS + \int_S Y dS/dtheta
        def f_non_linear(x):
            if x<self.d_min_hard:
                #raise Exception('minimal distance overeached')
                return np.inf
            else :
                return self.d_min_penalization*np.max((self.d_min_soft-x,0))**2/(1-np.max((self.d_min_soft-x,0))/(self.d_min_soft-self.d_min_hard))
        def grad_f_non_linear(x):
            if self.d_min_hard <x :
                if x>=self.d_min_soft:
                    return 0
                else:
                    c=self.d_min_soft-self.d_min_hard
                    y=self.d_min_soft-x
                    return self.d_min_penalization*(-c*y*(2*c - y))/((c - y)**2)
            else:
                #raise Exception('minimal distance overeached in gradient')
                return -np.inf
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
                vgradf[i,j]=grad_f_non_linear(dist_min[i,j])
                Y[i,j]=f_non_linear(dist_min[i,j])
        grad_d_min=T_min/dist_min[:,:,np.newaxis]
        X=-1*vgradf[:,:,np.newaxis]*grad_d_min
        return X,Y
    def non_linear_cost(d_min_hard,d_min_soft,d_min_penalization,dist,dS):
        assert(0<=d_min_hard and d_min_hard<d_min_soft)
        dist_min=np.amin(dist,axis=(0,3,4))
        def f_non_linear(x):
            if x<d_min_hard:
                return np.inf
                #raise Exception('minimal distance overeached')
            else :
                return d_min_penalization*np.max((d_min_soft-x,0))**2/(1-np.max((d_min_soft-x,0))/(d_min_soft-d_min_hard))
        vf = np.vectorize(f_non_linear)
        return np.einsum('ij,ij->',vf(dist_min),dS/dS.size)
    def non_linear_cost_gradient(d_min_hard,d_min_soft,d_min_penalization,T,dS):
        def grad_f_non_linear(x):
            if d_min_hard <x :
                if x>=d_min_soft:
                    return 0
                else:
                    c=d_min_soft-d_min_hard
                    y=d_min_soft-x
                    return d_min_penalization*(-c*y*(2*c - y))/((c - y)**2)
            else:
                return -np.inf
                #raise Exception('minimal distance overeached in gradient')
    def perimeter_cost(self,S,config):
        c0=float(config['optimization_parameters']['perim_c0'])
        c1=float(config['optimization_parameters']['perim_c1'])
        perim=self.Np*np.sum(S.dS)/S.npts
        def f_e(x):
            if 0 <=x and x <=c1:
                return np.max((x-c0,0))**2/(1- np.max((x-c0,0))/(c1-c0))
            else:
                return np.inf
                #raise Exception('infinite cost')
        return(f_e(perim))
    def perimeter_grad(self,S,dSdtheta,config):
        c0=float(config['optimization_parameters']['perim_c0'])
        c1=float(config['optimization_parameters']['perim_c1'])
        perim=self.Np*np.sum(S.dS)/S.npts
        grad_perim=self.Np*np.einsum('oij->o',dSdtheta)/S.npts
        def grad_f_e(x):
            if 0 <=x and x <=c1:
                if x<c0:
                    return 0
                else:
                    c=c1-c0
                    y=x-c0
                    return (c*y*(2*c - y))/((c - y)**2)
            else:
                return -np.inf
                #raise Exception('infinite cost in gradient')
        return grad_f_e(perim)*grad_perim
    def cost(self,paramS):
        S=Toroidal_surface(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        cost_regcoil_dic=cost_surface.cost_surface(self.config,S=S,Sp=self.Sp)
        cost_regcoil=cost_regcoil_dic['cost_B']+ self.lamb*cost_regcoil_dic['cost_J']
        T=tools.get_tensor_distance(S,self.Sp,self.rot_tensor)
        cost_distance=self.distance_cost(T,S)
        cost_perim=self.perimeter_cost(S,self.config)

        return np.array([cost_regcoil,cost_distance,cost_perim])
    def grad_cost(self,paramS):
        S=Toroidal_surface(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        result=self.compute_gradient_of(S=S)
        I_vector,I_matrix=result['I1']
        X,Y=self.gradient_cost_distance(result['T'],S)
        theta,dtildetheta,dtheta,dSdtheta=S.get_theta_pertubation()
        grad_regcoil=self.Np*(np.einsum('ija,oija,ij->o',I_vector,theta,S.dS/S.npts)+np.einsum('ijab,oijab,ij->o',I_matrix,dtildetheta,S.dS/S.npts))
        grad_distance=np.einsum('oijl,ijl,ij->o',theta,X,S.dS/S.npts)+np.einsum('ij,oij->o',Y,dSdtheta/S.npts)
        grad_perim=self.perimeter_grad(S,dSdtheta,self.config)
        return np.array([grad_regcoil,grad_distance,grad_perim])
