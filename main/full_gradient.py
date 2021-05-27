from shape_gradient import Shape_gradient
import numpy as np
import configparser
class full_gradient(Shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        super().__init__(self,config=config)
    def non_linear_cost(dmin_hard,dmin_soft,penalizarion_dmin,dist,dS):
        assert(0<=dmin_hard and dmin_hard<dmin_soft)
        dist_min=np.amin(dist,axis=(0,3,4))
        def f_non_linear(x):
            if x<dmin_hard:
                raise Exception('minimal distance overeached')
            else :
                return penalizarion_dmin*np.max((dmin_soft-x,0))**2/(1-np.max((dmin_soft-x,0))/(dmin_soft-dmin_hard))
        vf = np.vectorize(f_non_linear)
        return np.einsum('ij,ij->',vf(dist_min),dS/dS.size)
    def non_linear_cost_gradient(dmin_hard,dmin_soft,penalizarion_dmin,T,dist,dS):
        def grad_f_non_linear(x):
            if dmin_hard <x :
                if x>=dmin_soft:
                    return 0
                else:
                    c=dmin_soft-dmin_hard
                    y=dmin_soft-x
                    return penalizarion_dmin*(-c*y*(2*c - y))/((c - y)**2)
            else:
                raise Exception('minimal distance overeached in gradient')