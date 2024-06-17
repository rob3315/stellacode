import unittest
import numpy as np
import logging

from numpy.core.fromnumeric import shape

from src.costs.EM_shape_gradient import *
from src.costs.EM_cost import *
from src.costs.distance_shape_gradient import Distance_shape_gradient
from src.costs.perimeter_shape_gradient import Perimeter_shape_gradient
from src.costs.curvature_shape_gradient import Curvature_shape_gradient
from src.surface.surface_Fourier import Surface_Fourier
import src.tools as tools


class Test_numerical_gradient(unittest.TestCase):
    def test_EM_cost(self):
        """
        Test the numerical gradient of the EM cost function.

        This test verifies that the numerical gradient of the EM cost function
        matches the gradient computed by the shape gradient class.
        """
        # Set the random seed for reproducibility
        np.random.seed(2)

        # Create an instance of the shape gradient class
        shape_grad = EM_shape_gradient('config_file/config_small_debug.ini')

        # Create a surface using the parametrization from the shape gradient class
        S_parametrization = shape_grad.S_parametrization
        S1 = Surface_Fourier(S_parametrization, (17, 19), 3)

        # Set the step size for the numerical gradient
        eps = 1e-9

        # Compute the number of parameters in the parametrization
        ls = len(S_parametrization[0])

        # Generate a random perturbation vector
        perturb = (2 * np.random.random(2 * ls) - 1)

        # Create a new surface by applying the perturbation to the parametrization
        new_param = Surface_Fourier.change_param(
            S_parametrization, eps * perturb)
        S2 = Surface_Fourier(new_param, (17, 19), 3)

        # Compute the cost function for the initial and perturbed surfaces
        cost1, _ = shape_grad.cost(S1)
        cost2, _ = shape_grad.cost(S2)

        # Compute the gradient of the cost function w.r.t. the parametrization
        theta_pertubation = S1.get_theta_pertubation()
        grad = shape_grad.shape_gradient(S1, theta_pertubation)

        # Compute the numerical gradient
        dcost = np.einsum('a,a...->...', perturb, grad)

        # Compute the numerical gradient by finite differences
        dcost_num = (cost2 - cost1) / eps

        # Assert that the numerical and computed gradients are almost equal
        np.testing.assert_array_almost_equal(dcost, dcost_num, decimal=5)

    def test_distance_cost(self):
        """
        Test the numerical gradient of the distance cost function.

        This test verifies that the numerical gradient of the distance cost function
        matches the gradient computed by the shape gradient class.
        """
        # Set the random seed for reproducibility
        np.random.seed(2)

        # Create an instance of the shape gradient class
        shape_grad = Distance_shape_gradient(
            'config_file/config_small_debug.ini')

        # Create a surface using the parametrization from the shape gradient class
        S_parametrization = shape_grad.S_parametrization
        S1 = Surface_Fourier(S_parametrization, (17, 19), 3)

        # Set the step size for the numerical gradient
        eps = 1e-9

        # Compute the number of parameters in the parametrization
        ls = len(S_parametrization[0])

        # Generate a random perturbation vector
        perturb = (2 * np.random.random(2 * ls) - 1)

        # Create a new surface by applying the perturbation to the parametrization
        new_param = Surface_Fourier.change_param(
            S_parametrization, eps * perturb)
        S2 = Surface_Fourier(new_param, (17, 19), 3)

        # Compute the cost function for the initial and perturbed surfaces
        cost1, _ = shape_grad.cost(S1)
        cost2, _ = shape_grad.cost(S2)

        # Compute the gradient of the cost function w.r.t. the parametrization
        theta_pertubation = S1.get_theta_pertubation()
        grad = shape_grad.shape_gradient(S1, theta_pertubation)

        # Compute the numerical gradient
        dcost = np.einsum('a,a...->...', perturb, grad)

        # Compute the numerical gradient by finite differences
        dcost_num = (cost2 - cost1) / eps

        # Assert that the numerical and computed gradients are almost equal
        np.testing.assert_array_almost_equal(dcost, dcost_num, decimal=3)

    def test_perimeter_cost(self):
        """
        Test the numerical gradient of the perimeter cost function.

        This test verifies that the numerical gradient of the perimeter cost function
        matches the gradient computed by the shape gradient class.
        """
        # Set the random seed for reproducibility
        np.random.seed(2)

        # Create an instance of the shape gradient class
        shape_grad_EM = EM_shape_gradient('config_file/config_small_debug.ini')
        shape_grad = Perimeter_shape_gradient(
            'config_file/config_small_debug.ini')

        # Create a surface using the parametrization from the shape gradient class
        S_parametrization = shape_grad_EM.S_parametrization
        S1 = Surface_Fourier(S_parametrization, (17, 19), 3)

        # Set the step size for the numerical gradient
        eps = 1e-9

        # Compute the number of parameters in the parametrization
        ls = len(S_parametrization[0])

        # Generate a random perturbation vector
        perturb = (2 * np.random.random(2 * ls) - 1)

        # Create a new surface by applying the perturbation to the parametrization
        new_param = Surface_Fourier.change_param(
            S_parametrization, eps * perturb)
        S2 = Surface_Fourier(new_param, (17, 19), 3)

        # Compute the cost function for the initial and perturbed surfaces
        cost1, _ = shape_grad.cost(S1)
        cost2, _ = shape_grad.cost(S2)

        # Compute the gradient of the cost function w.r.t. the parametrization
        theta_pertubation = S1.get_theta_pertubation()
        grad = shape_grad.shape_gradient(S1, theta_pertubation)

        # Compute the numerical gradient
        dcost = np.einsum('a,a...->...', perturb, grad)

        # Compute the numerical gradient by finite differences
        dcost_num = (cost2 - cost1) / eps

        # Assert that the numerical and computed gradients are almost equal
        np.testing.assert_array_almost_equal(dcost, dcost_num, decimal=1)

    def test_curvature_cost(self):
        """
        Test the numerical gradient of the curvature cost function.

        This test verifies that the numerical gradient of the curvature cost function
        matches the gradient computed by the shape gradient class.
        """

        # Set the random seed for reproducibility
        np.random.seed(2)

        # Create an instance of the shape gradient class
        shape_grad_EM = EM_shape_gradient('config_file/config_full.ini')

        # Set the curvature regularization parameter
        shape_grad_EM.config['optimization_parameters']['curvature_c0'] = '6'

        # Create an instance of the curvature shape gradient class
        shape_grad = Curvature_shape_gradient(config=shape_grad_EM.config)

        # Load the CWS surface
        S_parametrization = Surface_Fourier.load_file(
            str(shape_grad.config['geometry']['path_CWS']))

        # Create a surface using the parametrization
        S1 = Surface_Fourier(
            S_parametrization, (shape_grad.ntheta_coil, shape_grad.nzeta_coil), shape_grad.Np)

        # Set the step size for the numerical gradient
        eps = 1e-9

        # Compute the number of parameters in the parametrization
        ls = len(S_parametrization[0])

        # Generate a random perturbation vector
        perturb = (2 * np.random.random(2 * ls) - 1)

        # Create a new surface by applying the perturbation to the parametrization
        new_param = Surface_Fourier.change_param(
            S_parametrization, eps * perturb)
        S2 = Surface_Fourier(
            new_param, (shape_grad.ntheta_coil, shape_grad.nzeta_coil), shape_grad.Np)

        # Compute the cost function for the initial and perturbed surfaces
        cost1, _ = shape_grad.cost(S1)
        cost2, _ = shape_grad.cost(S2)

        # Compute the gradient of the cost function w.r.t. the parametrization
        theta_pertubation = S1.get_theta_pertubation()
        grad = shape_grad.shape_gradient(S1, theta_pertubation)

        # Compute the numerical gradient
        dcost = np.einsum('a,a...->...', perturb, grad)

        # Compute the numerical gradient by finite differences
        dcost_num = (cost2 - cost1) / eps

        # Assert that the numerical and computed gradients are almost equal
        np.testing.assert_array_almost_equal(dcost, dcost_num, decimal=1)


if __name__ == '__main__':
    unittest.main()
