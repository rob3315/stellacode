import unittest
import numpy as np
import logging

from src.costs.EM_shape_gradient import *
from src.costs.EM_cost import *
from src.costs.distance_shape_gradient import Distance_shape_gradient
from src.costs.perimeter_shape_gradient import Perimeter_shape_gradient
from src.surface.surface_Fourier import Surface_Fourier
import src.tools as tools

class Test_numerical_gradient(unittest.TestCase):
    def test_EM_cost(self):
        np.random.seed(2)
        shape_grad=EM_shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad.S_parametrization
        S1=Surface_Fourier(S_parametrization,(17,19),3)
        eps=1e-9
        ls=len(S_parametrization[0])
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Surface_Fourier.change_param(S_parametrization, eps*perturb)
        S2=Surface_Fourier(new_param,(17,19),3)
        cost1=shape_grad.cost(S1)
        cost2=shape_grad.cost(S2)
        theta_pertubation=S1.get_theta_pertubation()
        grad=shape_grad.shape_gradient(S1,theta_pertubation)
        dcost=np.einsum('a,a...->...',perturb,grad)
        dcost_num=(cost2-cost1)/eps
        np.testing.assert_array_almost_equal(dcost,dcost_num,decimal=5)

    def test_distance_cost(self):
        np.random.seed(2)
        shape_grad_EM=EM_shape_gradient('config_file/config_small.ini')
        shape_grad=Distance_shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad_EM.S_parametrization
        S1=Surface_Fourier(S_parametrization,(17,19),3)
        eps=1e-9
        ls=len(S_parametrization[0])
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Surface_Fourier.change_param(S_parametrization, eps*perturb)
        S2=Surface_Fourier(new_param,(17,19),3)
        cost1=shape_grad.cost(S1)
        cost2=shape_grad.cost(S2)
        theta_pertubation=S1.get_theta_pertubation()
        grad=shape_grad.shape_gradient(S1,theta_pertubation)
        dcost=np.einsum('a,a...->...',perturb,grad)
        dcost_num=(cost2-cost1)/eps
        np.testing.assert_array_almost_equal(dcost,dcost_num,decimal=3)
    def test_perimeter_cost(self):
        np.random.seed(2)
        shape_grad_EM=EM_shape_gradient('config_file/config_small.ini')
        shape_grad=Perimeter_shape_gradient('config_file/config_small.ini')
        S_parametrization=shape_grad_EM.S_parametrization
        S1=Surface_Fourier(S_parametrization,(17,19),3)
        eps=1e-9
        ls=len(S_parametrization[0])
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Surface_Fourier.change_param(S_parametrization, eps*perturb)
        S2=Surface_Fourier(new_param,(17,19),3)
        cost1=shape_grad.cost(S1)
        cost2=shape_grad.cost(S2)
        theta_pertubation=S1.get_theta_pertubation()
        grad=shape_grad.shape_gradient(S1,theta_pertubation)
        dcost=np.einsum('a,a...->...',perturb,grad)
        dcost_num=(cost2-cost1)/eps
        np.testing.assert_array_almost_equal(dcost,dcost_num,decimal=1)

if __name__ == '__main__':
    unittest.main()