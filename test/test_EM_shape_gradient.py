import unittest
import numpy as np
import logging

from src.costs.EM_shape_gradient import *
from src.costs.EM_cost import *
from src.surface.surface_Fourier import *

class Test_EM_shape_gradient(unittest.TestCase):
    #@unittest.skip('debugging')
    def test_compatible_cost(self):
        np.random.seed(2)
        shape_grad=EM_shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad.S_parametrization
        S_parametrization2=Surface_Fourier.load_file('data/li383/cws.txt')
        S=Surface_Fourier(S_parametrization,(17,19),3)
        Sp_parametrization=Surface_Fourier.load_file('data/li383/plasma_surf.txt')
        Sp=Surface_Fourier(Sp_parametrization,(16,18),3)
        cost_from_shape_opti=shape_grad.compute_gradient_df(S_parametrization)
        cost_from_cost_surface=EM_cost(shape_grad.config)
        cost_from_cost_surface2=EM_cost_dask_old(shape_grad.config,S,Sp)
        np.testing.assert_almost_equal(cost_from_cost_surface['cost'],cost_from_shape_opti['cost'])
        np.testing.assert_almost_equal(cost_from_cost_surface2['cost'],cost_from_shape_opti['cost'])
    def test_EM_shape_gradient_df(self):
        np.random.seed(2)
        #from dask.distributed import Client
        #client = Client(processes=False,
        #                n_workers=5, threads_per_worker=4)
        #client.scheduler_info()['services']
        shape_grad=EM_shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad.S_parametrization
        eps=1e-6
        ls=len(S_parametrization[0])
        #we compute the shape derivative
        result1=shape_grad.compute_gradient_df(S_parametrization)
        
        #we apply one pertubation
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Surface_Fourier.change_param(S_parametrization, eps*perturb)
        result2=shape_grad.compute_gradient_df(new_param)

        #we list the element to test
        quantities=['LS_matrix','RHS','j_S_partial','cost_J','cost_B','cost']
        gradient_quantities=['dLS_matrix_dtheta','dRHS_dtheta','dj_S_partial_dtheta','dcost_J_dtheta','dcost_B_dtheta','EM_shape_gradient']
        decimal_precision=[8,7,-4,-13,2,2]
        for quantity,grad,precision in zip(quantities,gradient_quantities,decimal_precision):
            print(quantity,grad)
            q1,grad_q1 = result1[quantity],result1[grad]
            q2= result2[quantity]

            dq_num=(q2-q1)/eps
            dq=np.einsum('a,a...->...',perturb,grad_q1)
            np.testing.assert_array_almost_equal(dq,dq_num,decimal=precision)
            #print(dq.shape)
    def test_EM_shape_gradient_of(self):
        np.random.seed(2)
        #from dask.distributed import Client
        #client = Client()
        #print(client.scheduler_info()['services'])
        shape_grad=EM_shape_gradient('config_file/config_small.ini')
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
        #II=2*lamb*contract('p,pq,oq',j_S,Qj[:,2:],dj_S_partial_dtheta)+2*contract('ij,ij,qij,oq->o',(contract('abc,a->bc',LS,j_S)-BT),shape_grad.Sp.dS/shape_grad.Sp.npts,LS[2:],dj_S_partial_dtheta)
        #np.testing.assert_array_almost_equal(lamb*result1['dcost_J_dtheta']/shape_grad.Np,I+II)
        #np.testing.assert_array_almost_equal(result1['dcost_B_dtheta']/shape_grad.Np,I+II)
        np.testing.assert_array_almost_equal(result1['EM_shape_gradient']/shape_grad.Np,I)
        #III=2*contract('p,pq,oq',result2['h'].compute(),Qj[2:,2:],dj_S_partial_dtheta)
        result2=shape_grad.compute_gradient_of(S_parametrization)

        S=Surface_Fourier(S_parametrization,(shape_grad.ntheta_coil,shape_grad.nzeta_coil),shape_grad.Np)
        theta_pertubation=S.get_theta_pertubation()
        theta=theta_pertubation['theta']
        dtildetheta=theta_pertubation['dtildetheta']
        X1,X2=result2['I1']
        gradient=np.einsum('ija,oija,ij->o',X1,theta,S.dS/S.npts)+np.einsum('ijab,oijab,ij->o',X2,dtildetheta,S.dS/S.npts)
        np.testing.assert_array_almost_equal(I,gradient)

if __name__ == '__main__':
    unittest.main()