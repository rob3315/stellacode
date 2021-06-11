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
        theta_pertubation=S1.get_theta_pertubation()
        grad_result=shape_grad.curvature_derivative(S1,theta_pertubation)
        # We test the first fundamental form derivative
        numerical_gradient=[(S2.I[0]-S1.I[0])/eps,(S2.I[1]-S1.I[1])/eps,(S2.I[2]-S1.I[2])/eps]
        th_gradient=grad_result['dI']
        decimal_lst=[3,3,3]
        for num_elt,th_elt,decim in zip(numerical_gradient,th_gradient,decimal_lst):
            dth_elt=np.einsum('a,a...->...',perturb,th_elt)
            np.testing.assert_array_almost_equal(num_elt,dth_elt,decimal=decim)

if __name__ == '__main__':
    unittest.main()