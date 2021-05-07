import os
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
import unittest
from shape_gradient import *
import numpy as np
import logging

class Test_shape_gradient(unittest.TestCase):
    #@unittest.skip('debugging')
    def test_shape_gradient_df(self):
        np.random.seed(2)
        from dask.distributed import Client
        client = Client(processes=False,
                        n_workers=5, threads_per_worker=4)
        client.scheduler_info()['services']
        shape_grad=Shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad.S_parametrization
        eps=1e-6
        ls=len(S_parametrization[0])
        #we compute the shape derivative
        result1=shape_grad.compute_gradient_df(S_parametrization)
        
        #we apply one pertubation
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Toroidal_surface.change_param(S_parametrization, eps*perturb)
        result2=shape_grad.compute_gradient_df(new_param)

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
            #print(dq.shape)
    def test_shape_gradient_of(self):
        np.random.seed(2)
        from dask.distributed import Client
        client = Client()
        print(client.scheduler_info()['services'])
        shape_grad=Shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad.S_parametrization
        eps=1e-6
        #we compute the shape derivative
        result1=shape_grad.compute_gradient_df(S_parametrization)
        #we check the computation of I
        lamb=shape_grad.lamb
        j_S=np.concatenate(([shape_grad.net_poloidal_current_Amperes,shape_grad.net_toroidal_current_Amperes],result1['j_S_partial']))
        Qj=result1['Qj']
        dQj=result1['dQj']
        LS=result1['LS']
        BT=-result1['array_bnorm']
        dLSdtheta=result1['dLSdtheta']
        dj_S_partial_dtheta=result1['dj_S_partial_dtheta']
        I=lamb*contract('i,oij,j->o',j_S,dQj,j_S)+2*contract('ij,ij,opij,p->o',(contract('abc,a->bc',LS,j_S)-BT),shape_grad.Sp.dS,dLSdtheta,j_S)/(shape_grad.Sp.nbpts[0]*shape_grad.Sp.nbpts[1])
        II=2*lamb*contract('p,pq,oq',j_S,Qj[:,2:],dj_S_partial_dtheta)+2*contract('ij,ij,qij,oq->o',(contract('abc,a->bc',LS,j_S)-BT),shape_grad.Sp.dS,LS[2:],dj_S_partial_dtheta)/(shape_grad.Sp.nbpts[0]*shape_grad.Sp.nbpts[1])
        #np.testing.assert_array_almost_equal(lamb*result1['dcost_J_dtheta']/shape_grad.Np,I+II)
        #np.testing.assert_array_almost_equal(result1['dcost_B_dtheta']/shape_grad.Np,I+II)
        np.testing.assert_array_almost_equal(result1['shape_gradient']/shape_grad.Np,I+II)
        result2=shape_grad.compute_gradient_of(S_parametrization)
        np.testing.assert_array_almost_equal(result1['j_S_partial'],result2['j_S_partial'])
        print('done')
        

        pass


if __name__ == '__main__':
    print(os.getcwd())
    unittest.main()