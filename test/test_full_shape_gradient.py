import unittest
import numpy as np
import logging
from src.costs.full_shape_gradient import Full_shape_gradient

class Test_full_gradient(unittest.TestCase):
    def test_full_numerical_gradient(self):
        np.random.seed(25)
        eps=1e-9
        full_grad=Full_shape_gradient(path_config_file='config_file/config_small.ini')
        param=full_grad.init_param
        cost1=full_grad.cost(param)
        perturb=(2*np.random.random(len(param))-1)
        new_param=param+ eps*perturb
        cost2=full_grad.cost(new_param)
        #We compute the gradient
        grad=full_grad.shape_grad(param)
        #we compare the gradients
        grad_num=(cost2-cost1)/eps
        grad_th=grad@perturb
        np.testing.assert_almost_equal(np.max(np.abs(grad_num-grad_th)),0,decimal=0)

if __name__ == '__main__':
    unittest.main()