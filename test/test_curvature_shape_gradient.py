import unittest
import numpy as np
import logging

from src.costs.curvature_shape_gradient import *
from src.surface.surface_Fourier import *

class Test_EM_shape_gradient(unittest.TestCase):
    #@unittest.skip('debugging')
    def test_curvature_derivative(self):
        np.random.seed(2)
        shape_grad=Curvature_shape_gradient('config_file/config_full.ini')
        S_parametrization=Surface_Fourier.load_file(str(shape_grad.config['geometry']['path_CWS']))
        S1=Surface_Fourier(S_parametrization,(shape_grad.ntheta_coil,shape_grad.nzeta_coil),shape_grad.Np)
        eps=1e-9
        ls=len(S_parametrization[0])
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Surface_Fourier.change_param(S_parametrization, eps*perturb)
        S2=Surface_Fourier(new_param,(shape_grad.ntheta_coil,shape_grad.nzeta_coil),shape_grad.Np)
        theta_pertubation=S1.get_theta_pertubation(compute_curvature=True)
        grad_result=shape_grad.curvature_derivative(S1,theta_pertubation)
        # We test the first fundamental form derivative
        numerical_gradient_1=[(S2.I[0]-S1.I[0])/eps,(S2.I[1]-S1.I[1])/eps,(S2.I[2]-S1.I[2])/eps]
        th_gradient_I=grad_result['dI']
        decimal_lst=[3,3,3]
        for num_elt,th_elt,decim in zip(numerical_gradient_1,th_gradient_I,decimal_lst):
            dth_elt=np.einsum('a,a...->...',perturb,th_elt)
            np.testing.assert_array_almost_equal(num_elt,dth_elt,decimal=decim)
        numerical_gradient_d2=[(S2.dpsi_uu-S1.dpsi_uu)/eps,(S2.dpsi_uv-S1.dpsi_uv)/eps,(S2.dpsi_vv-S1.dpsi_vv)/eps]
        th_gradient_d2=theta_pertubation['d2theta']
        decimal_lst=[3,3,3]
        for num_elt,th_elt,decim in zip(numerical_gradient_d2,th_gradient_d2,decimal_lst):
            dth_elt=np.einsum('a,aijl->lij',perturb,th_elt)
            np.testing.assert_array_almost_equal(num_elt,dth_elt,decimal=decim)
        numerical_gradientII=[(S2.II[0]-S1.II[0])/eps,(S2.II[1]-S1.II[1])/eps,(S2.II[2]-S1.II[2])/eps]
        th_gradientII=grad_result['dII']
        decimal_lst=[2,2,2]
        for num_elt,th_elt,decim in zip(numerical_gradientII,th_gradientII,decimal_lst):
            dth_elt=np.einsum('a,a...->...',perturb,th_elt)
            np.testing.assert_array_almost_equal(num_elt,dth_elt,decimal=decim)
        numerical_gradient_curvature=[(S2.K-S1.K)/eps,(S2.H-S1.H)/eps]
        th_gradient_curvature=[grad_result['dK'],grad_result['dH']]
        decimal_lst=[2,3]
        for num_elt,th_elt,decim in zip(numerical_gradient_curvature,th_gradient_curvature,decimal_lst):
            dth_elt=np.einsum('a,a...->...',perturb,th_elt)
            np.testing.assert_array_almost_equal(num_elt,dth_elt,decimal=decim)
        numerical_gradient_principle_curvature=[(S2.principles[0]-S1.principles[0])/eps,(S2.principles[1]-S1.principles[1])/eps]
        th_gradient_principle_curvature=[grad_result['dPmax'],grad_result['dPmin']]
        decimal_lst=[1,2]
        for num_elt,th_elt,decim in zip(numerical_gradient_principle_curvature,th_gradient_principle_curvature,decimal_lst):
            dth_elt=np.einsum('a,a...->...',perturb,th_elt)
            np.testing.assert_array_almost_equal(num_elt,dth_elt,decimal=decim)
        

if __name__ == '__main__':
    unittest.main()