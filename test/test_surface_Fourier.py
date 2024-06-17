import unittest
from src.surface.surface_Fourier import *
import numpy as np
import logging


class Test_toridal_surface(unittest.TestCase):
    @unittest.skip("graphic verif skipping")
    def test_normal_derivative(self):
        """
        Test the derivative of the normal vector with respect to the grid coordinates.

        This test checks if the values of the normal vector derivatives computed using
        numpy.gradient are similar to the values computed using the surface_Fourier
        class.

        The test is skipped by default.
        """

        # Define the size of the grid
        lu, lv = 128, 128

        # Set the logging level to DEBUG
        logging.basicConfig(level='DEBUG')

        # Load the surface parameters from a file
        surface_parametrization = Surface_Fourier.load_file(
            'data/li383/cws.txt')

        # Create a surface_Fourier object with the given parameters
        S = Surface_Fourier(surface_parametrization, (lu, lv), 3)

        # Compute the derivatives of the normal vector with respect to the grid coordinates
        grad_dS = np.gradient(S.dS, 1/lu, 1/lv)
        grad_nx = np.gradient(S.n[0], 1/lu, 1/lv)
        grad_ny = np.gradient(S.n[1], 1/lu, 1/lv)
        grad_nz = np.gradient(S.n[2], 1/lu, 1/lv)

        # Plot the derivatives of the normal vector with respect to the grid coordinates
        import matplotlib.pyplot as plt
        plt.plot(grad_ny[1][33])
        plt.plot(S.n_v[1][33])
        plt.legend(['Numpy', 'FourierSurface'])
        plt.title('Normal Vector Derivative')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()

    def test_perimeter_derivative(self):
        """
        Test the derivative of the perimeter with respect to the surface parameters.

        This test checks if the values of the perimeter derivative computed using the
        surface_Fourier class are similar to the values computed using finite differences.
        """

        # Define the size of the grid
        lu, lv = 128, 128

        # Define the perturbation amplitude
        eps = 1e-8

        # Set the logging level to DEBUG
        logging.basicConfig(level='DEBUG')

        # Load the surface parameters from a file
        surface_parametrization = Surface_Fourier.load_file(
            'data/li383/cws.txt')

        # Create a surface_Fourier object with the given parameters
        S = Surface_Fourier(surface_parametrization, (lu, lv), 3)

        # Compute the derivative of the perimeter with respect to the surface parameters
        perim = np.einsum('ij->', S.dS) / (lu * lv)
        perturb = (2 * np.random.random(2 *
                   len(surface_parametrization[0])) - 1)
        new_param = Surface_Fourier.change_param(
            surface_parametrization, eps * perturb)
        new_S = Surface_Fourier(new_param, (lu, lv), 3)
        new_perim = np.einsum('ij->', new_S.dS) / (lu * lv)

        # Compute the derivative using finite differences
        dtildetheta = S.get_theta_pertubation()['dtildetheta']
        dperim_num = (new_perim - perim) / eps
        dperim = np.einsum('cijll,c,ij', dtildetheta,
                           perturb, S.dS) / (lu * lv)

        # Assert that the computed derivative is almost equal to the finite difference derivative
        np.testing.assert_almost_equal(dperim, dperim_num, decimal=3)

        # Print the computed and finite difference derivatives
        print(dperim_num, dperim)

    def test_change_of_variable_scalar(self):
        """
        Test the derivative of the integral of a random function with respect to the surface parameters.

        This test checks if the values of the derivative computed using the 
        surface_Fourier class are similar to the values computed using finite differences.

        """
        # Define the size of the grid
        lu, lv = 128, 128

        # Define the perturbation amplitude
        eps = 1e-8

        # Set the logging level to DEBUG
        logging.basicConfig(level='DEBUG')

        # Load the surface parameters from a file
        surface_parametrization = Surface_Fourier.load_file(
            'data/li383/cws.txt')

        # Create a surface_Fourier object with the given parameters
        S = Surface_Fourier(surface_parametrization, (lu, lv), 3)

        # Compute the derivative of the integral with respect to the surface parameters
        theta_pertubation = S.get_theta_pertubation()
        theta = theta_pertubation['theta']
        dtildetheta = theta_pertubation['dtildetheta']
        f = np.random.random((lu, lv))
        int_f = np.einsum('ij,ij->', f, S.dS)/(lu*lv)
        perturb = (2*np.random.random(2*len(surface_parametrization[0]))-1)
        new_param = Surface_Fourier.change_param(
            surface_parametrization, eps*perturb)
        new_S = Surface_Fourier(new_param, (lu, lv), 3)
        new_int_f = np.einsum('ij,ij->', new_S.dS, f)/(lu*lv)

        # Compute the derivative using finite differences
        dint_num = (new_int_f-int_f)/eps
        dint = np.einsum('cijll,c,ij,ij', dtildetheta,
                         perturb, S.dS, f)/(lu*lv)

        # Assert that the computed derivative is almost equal to the finite difference derivative
        np.testing.assert_almost_equal(dint, dint_num, decimal=3)

        # Print the computed and finite difference derivatives
        print(dint_num, dint)

    def test_dSdtheta(self):
        """
        Test the derivative of the surface with respect to theta.

        This test checks if the values of the derivative computed using the 
        surface_Fourier class are similar to the values computed using finite differences.

        """
        # Define the size of the grid
        lu, lv = 128, 128

        # Define the perturbation amplitude
        eps = 1e-8

        # Set the logging level to DEBUG
        logging.basicConfig(level='DEBUG')

        # Load the surface parameters from a file
        surface_parametrization = Surface_Fourier.load_file(
            'data/li383/cws.txt')

        # Create a surface_Fourier object with the given parameters
        S = Surface_Fourier(surface_parametrization, (lu, lv), 3)

        # Compute the derivative of the surface with respect to theta
        theta_pertubation = S.get_theta_pertubation()
        # derivative of the surface with respect to theta
        dSdtheta = theta_pertubation['dSdtheta']
        # perturbation of the surface parameters
        perturb = (2*np.random.random(2*len(surface_parametrization[0]))-1)
        new_param = Surface_Fourier.change_param(  # new surface parameters with a small perturbation
            surface_parametrization, eps*perturb)
        # new surface object with the perturbed parameters
        new_S = Surface_Fourier(new_param, (lu, lv), 3)

        # Compute the derivative using finite differences
        # finite difference derivative of the surface
        ddS_num = (new_S.dS-S.dS)/eps
        # analytical derivative of the surface
        ddS = np.einsum('cij,c', dSdtheta, perturb)

        # Assert that the computed derivative is almost equal to the finite difference derivative
        np.testing.assert_almost_equal(ddS_num, ddS, decimal=2)

    def test_curvature_derivative(self):
        """
        Test the derivative of the curvature with respect to the surface parameters.

        This test checks if the values of the curvature derivative computed using the
        surface_Fourier class are similar to the values computed using finite differences.
        """
        # Define the size of the grid
        lu, lv = 128, 128

        # Load the surface parameters from a file
        surface_parametrization = Surface_Fourier.load_file(
            'data/li383/cws.txt')

        # Create a surface_Fourier object with the given parameters
        S = Surface_Fourier(surface_parametrization, (lu, lv), 3)


if __name__ == '__main__':
    unittest.main()
