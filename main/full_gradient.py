from shape_gradient import Shape_gradient
import numpy as np
import configparser
class Full_gradient(Shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        super().__init__(self,config=config)
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
                raise Exception('minimal distance overeached')
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
                raise Exception('minimal distance overeached in gradient')
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
                raise Exception('minimal distance overeached')
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
                raise Exception('minimal distance overeached in gradient')
    