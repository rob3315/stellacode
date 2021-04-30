import os
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
import unittest
from regcoil import *
from shape_gradient_dask import *
import numpy as np
import logging

class Test_shape_opti(unittest.TestCase):
    def test_shape_gradient(self):
        from dask.distributed import Client
        client = Client(processes=False,
                        n_workers=5, threads_per_worker=10)
        client.scheduler_info()['services']
        shape_grad=Shape_gradient_dask('code/config_small.ini')
        S_parametrization=shape_grad.S_parametrization
        shape_grad.compute_gradient(S_parametrization)
        eps=1e-6
        ls=len(S_parametrization[0])
        #we compute the shape derivative
        result1=shape_grad.compute_gradient(S_parametrization)
        
        #we apply one pertubation
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Toroidal_surface.change_param(S_parametrization, eps*perturb)
        result2=shape_grad.compute_gradient(new_param)

        #we list the element to test
        quantities=['LS_matrix','RHS','j_S_partial','cost_J','cost_B','cost']
        gradient_quantities=['dLS_matrix_dtheta','dRHS_dtheta','dj_S_partial_dtheta','dcost_J_dtheta','dcost_B_dtheta','shape_gradient']
        decimal_precision=[8,7,-4,-13,2,2]
        for quantity,grad,precision in zip(quantities,gradient_quantities,decimal_precision):
            print(quantity,grad)
            q1,grad_q1 = result1[quantity],result1[grad]
            q2= result2[quantity]

            dq_num=(q2-q1)/eps
            dq=np.einsum('a,a...->...',perturb,grad_q1)
            np.testing.assert_array_almost_equal(dq,dq_num,decimal=precision)
            print(dq.shape)


if __name__ == '__main__':
    print(os.getcwd())
    unittest.main()