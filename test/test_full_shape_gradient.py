import unittest
import numpy as np
import logging
from src.costs.full_shape_gradient import Full_shape_gradient


class Test_full_gradient(unittest.TestCase):
    def test_full_numerical_gradient(self):
        """
        Tests the numerical gradient of the full gradient cost function.

        This test checks if the numerical gradient computed by the test method
        is almost equal to the gradient computed using the autograd library.
        The test is performed for a randomly generated set of parameters.
        """

        # Set the random seed for reproducibility
        np.random.seed(25)

        # Define the perturbation value
        eps = 1e-9

        # Create an instance of the full gradient cost function
        full_grad = Full_shape_gradient(
            path_config_file='config_file/config_small_debug.ini')

        # Get the initial parameters
        param = full_grad.init_param

        # Get the cost at the initial parameters
        cost1 = full_grad.cost(param)

        # Generate a random perturbation vector
        perturb = (2 * np.random.random(len(param)) - 1)

        # Compute the new parameters by adding the perturbation
        new_param = param + eps * perturb

        # Get the cost at the new parameters
        cost2 = full_grad.cost(new_param)

        # Compute the gradient of the cost function using the shape_gradient method
        grad = full_grad.shape_gradient(param)

        # Compute the numerical gradient
        grad_num = (cost2 - cost1) / eps

        # Compute the gradient using matrix multiplication
        grad_th = grad @ perturb

        # Assert that the numerical gradient is almost equal to the computed gradient
        np.testing.assert_almost_equal(
            np.max(np.abs(grad_num - grad_th)), 0, decimal=0)


if __name__ == '__main__':
    unittest.main()
