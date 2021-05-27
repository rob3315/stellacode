import os
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
import unittest
from full_gradient import *
import numpy as np
import logging

class Test_full_gradient(unittest.TestCase):
    #@unittest.skip('debugging')
    def test_auxiliary_cost(self):
        np.random.seed(2)
        full_grad=Full_gradient('config_file/config.ini')
        S_parametrization=full_grad.S_parametrization
        eps=1e-6
        ls=len(S_parametrization[0])
        #we compute the shape derivative
        result1=full_grad.compute_gradient_of(S_parametrization)
        
        #we apply one pertubation
        perturb=(2*np.random.random(2*ls)-1)
        dist=np.linalg.norm(T,axis=-1)
        indexes=np.argmin(np.reshape(dist,(3,64,64,-1)),axis=-1)
        aux= lambda  x : np.unravel_index(x,(64,64))
        v_aux=np.vectorize(aux)
        indexes_full1,indexes_full2=v_aux(indexes)
        

        #new_param=Toroidal_surface.change_param(S_parametrization, eps*perturb)
        #result2=shape_grad.compute_gradient_df(new_param)

        # #we list the element to test
        # quantities=['LS_matrix','RHS','j_S_partial','cost_J','cost_B','cost']
        # gradient_quantities=['dLS_matrix_dtheta','dRHS_dtheta','dj_S_partial_dtheta','dcost_J_dtheta','dcost_B_dtheta','shape_gradient']
        # decimal_precision=[8,7,-4,-13,2,2]
        # for quantity,grad,precision in zip(quantities,gradient_quantities,decimal_precision):
        #     print(quantity,grad)
        #     q1,grad_q1 = result1[quantity],result1[grad]
        #     q2= result2[quantity]

        #     dq_num=(q2-q1)/eps
        #     dq=np.einsum('a,a...->...',perturb,grad_q1)
        #     np.testing.assert_array_almost_equal(dq,dq_num,decimal=precision)
        #     #print(dq.shape)

if __name__ == '__main__':
    print(os.getcwd())
    unittest.main()