import unittest
import numpy as np
import logging
#from full_gradient import *
#import tools
#from toroidal_surface import *

class Test_full_gradient(unittest.TestCase):
    #@unittest.skip('debugging')
    def test_distance_cost(self):
        np.random.seed(2)
        full_grad=Full_gradient('config_file/config.ini')
        S_parametrization=full_grad.S_parametrization
        eps=1e-9
        ls=len(S_parametrization[0])
        #we compute the shape derivative
        S=Toroidal_surface(S_parametrization,(full_grad.ntheta_coil,full_grad.nzeta_coil),full_grad.Np)
        T=tools.get_tensor_distance(S,full_grad.Sp,full_grad.rot_tensor)
        theta,dtildetheta,dtheta,dSdtheta=S.get_theta_pertubation()
        cost1=full_grad.distance_cost(T,S)

        #we apply one pertubation
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Toroidal_surface.change_param(S_parametrization, eps*perturb)
        S2=Toroidal_surface(new_param,(full_grad.ntheta_coil,full_grad.nzeta_coil),full_grad.Np)
        T2=tools.get_tensor_distance(S2,full_grad.Sp,full_grad.rot_tensor)
        cost2=full_grad.distance_cost(T2,S2)
        #We compute the gradient
        X,Y=full_grad.gradient_cost_distance(T,S)
        #we compare the gradients
        grad_cost_num=(cost2-cost1)/eps
        grad_cost_th=np.einsum('oijl,ijl,o,ij->',theta,X,perturb,S.dS/S.npts)+np.einsum('ij,o,oij',Y,perturb,dSdtheta/S.npts)
        np.testing.assert_almost_equal(np.max(np.abs(grad_cost_num-grad_cost_th)),0,decimal=4)

if __name__ == '__main__':
    unittest.main()