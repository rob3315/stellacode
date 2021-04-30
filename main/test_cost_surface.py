import unittest
from cost_surface import *
import logging
import configparser
class Test_cost_surface(unittest.TestCase):

    def test_no_dimension_error(self):
        ###just check that all operations respects dimensions
        path_config_file='config_file/config_test_dim.ini'
        config = configparser.ConfigParser()
        config.read(path_config_file)
        cost_surface(config)
    def test_compare_to_regcoil(self):
        path_config_file='config_file/config.ini'
        config = configparser.ConfigParser()
        config.read(path_config_file)
        #data from laplace_code
        err_max_B=[1.26945824E-02,2.67485601E-01,5.42332579E-02,1.73186188E-02]
        max_j=[9.15419611E+07,3.94359068E+06,7.35442879E+06,1.52081420E+07]
        cost_B=[3.49300835E-04,1.36627491E-01,5.56293801E-03,4.67141080E-04]
        cost_J=[8.96827408E+15,1.00021438E+14,1.42451562E+14,6.65195111E+14]
        for index,lamb in enumerate([0,1.2e-14,2.5e-16,5.1e-19]):
            config['other']['lamb']=str(lamb)
            cost_surface_output=cost_surface(config)
            for key,value in cost_surface_output.items():
                print('{} : {:e}'.format(key,value))
            np.testing.assert_almost_equal(cost_surface_output['err_max_B'],err_max_B[index])
            np.testing.assert_almost_equal(cost_surface_output['max_j'],max_j[index],decimal=0)
            np.testing.assert_almost_equal(cost_surface_output['cost_B'],cost_B[index])
            np.testing.assert_almost_equal(cost_surface_output['cost_J'],cost_J[index],decimal=-8)
if __name__ == '__main__':
    unittest.main()