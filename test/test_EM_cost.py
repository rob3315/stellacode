import unittest
import logging
import configparser

from src.costs.EM_cost import *


class Test_EM_cost(unittest.TestCase):

    def test_no_dimension_error(self):
        """
        Test if the EM_cost function raises an error when dimensions are not respected.

        This test checks if all operations within the EM_cost function respect the dimensions.
        It reads a config file that has dimensions defined in it and passes it to the EM_cost function.
        If the function raises an error, it means that the dimensions are not respected.
        """
        # Define the path to the config file
        path_config_file = 'config_file/config_test_dim.ini'

        # Read the config file
        config = configparser.ConfigParser()
        config.read(path_config_file)

        # Call the EM_cost function with the config
        EM_cost(config)

    def test_compare_to_regcoil(self):
        """
        Test the comparison of the output of the EM_cost function with the data from the regcoil code.

        This test compares the output of the EM_cost function with the data from the regcoil code.
        It reads a config file that has the different values for lambda, and for each lambda value,
        it calculates the output of the EM_cost function and compares it with the data from the regcoil code.

        """
        # Define the path to the config file
        path_config_file = 'config_file/config.ini'

        # Read the config file
        config = configparser.ConfigParser()
        config.read(path_config_file)

        # Define the data from the regcoil code
        err_max_B = [1.26945824E-02, 2.67485601E-01,
                     5.42332579E-02, 1.73186188E-02]
        max_j = [9.15419611E+07, 3.94359068E+06,
                 7.35442879E+06, 1.52081420E+07]
        cost_B = [3.49300835E-04, 1.36627491E-01,
                  5.56293801E-03, 4.67141080E-04]
        cost_J = [8.96827408E+15, 1.00021438E+14,
                  1.42451562E+14, 6.65195111E+14]

        # Loop over the different lambda values
        for index, lamb in enumerate([0, 1.2e-14, 2.5e-16, 5.1e-19]):
            # Set the lambda value in the config file
            config['other']['lamb'] = str(lamb)

            # Calculate the output of the EM_cost function
            EM_cost_output = EM_cost(config)

            # Compare the output with the data from the regcoil code
            np.testing.assert_almost_equal(
                EM_cost_output['err_max_B'], err_max_B[index])
            np.testing.assert_almost_equal(
                EM_cost_output['max_j'], max_j[index], decimal=-1)
            np.testing.assert_almost_equal(
                EM_cost_output['cost_B'], cost_B[index])
            np.testing.assert_almost_equal(
                EM_cost_output['cost_J'], cost_J[index], decimal=-9)


if __name__ == '__main__':
    unittest.main()
