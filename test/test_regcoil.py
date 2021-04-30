import unittest
from regcoil import *
import logging
class Test_regcoil(unittest.TestCase):

    def test_no_dimension_error(self):
        ###just check that all operations respects dimensions
        ntheta_plasma = 61
        ntheta_coil   = 62
        nzeta_plasma = 63
        nzeta_coil   = 64
        mpol_coil  = 7
        ntor_coil  = 5
        Np=4 #symmetry
        net_poloidal_current_Amperes = 11884578.094260072/Np
        net_toroidal_current_Amperes = 0.#0.3#/(2*np.pi)
        lamb=0
        curpol=4.9782004309255496# convention for the bnorm data, 
        path_bnorm='code/data/li383/bnorm.txt'
        path_cws='code/data/li383/cws.txt'
        path_plasma='code/data/li383/plasma_surf.txt'
        regcoil_param=Regcoil_param(lamb,Np,path_cws, path_plasma,path_bnorm,curpol,net_poloidal_current_Amperes,net_toroidal_current_Amperes, ntheta_plasma = ntheta_plasma,ntheta_coil = ntheta_coil,nzeta_plasma = nzeta_plasma, nzeta_coil = nzeta_coil, mpol_coil = mpol_coil, ntor_coil = ntor_coil)
        regcoil(regcoil_param)
    def test_compare_to_regcoil(self):
        #data from my laplace_code
        err_max_B=[1.26945824E-02,2.67485601E-01,5.42332579E-02,1.73186188E-02]
        max_j=[9.15419611E+07,3.94359068E+06,7.35442879E+06,1.52081420E+07]
        cost_B=[3.49300835E-04,1.36627491E-01,5.56293801E-03,4.67141080E-04]
        cost_J=[8.96827408E+15,1.00021438E+14,1.42451562E+14,6.65195111E+14]
        for index,lamb in enumerate([0,1.2e-14,2.5e-16,5.1e-19]):
            logging.basicConfig(level=logging.DEBUG)
            ntheta_plasma = 64
            ntheta_coil   = 64
            nzeta_plasma = 64
            nzeta_coil   = 64
            mpol_coil  = 8
            ntor_coil  = 8
            Np=3 #symmetry
            net_poloidal_current_Amperes = 11884578.094260072/Np
            net_toroidal_current_Amperes = 0.#0.3#/(2*np.pi)
            curpol=4.9782004309255496# convention for the bnorm data, 
            path_bnorm='code/data/li383/bnorm.txt'
            path_cws='code/data/li383/cws.txt'
            path_plasma='code/data/li383/plasma_surf.txt'
            regcoil_param=Regcoil_param(lamb,Np,path_cws, path_plasma,path_bnorm,curpol,net_poloidal_current_Amperes,net_toroidal_current_Amperes, ntheta_plasma = ntheta_plasma,ntheta_coil = ntheta_coil,nzeta_plasma = nzeta_plasma, nzeta_coil = nzeta_coil, mpol_coil = mpol_coil, ntor_coil = ntor_coil)
            regcoil_output=regcoil(regcoil_param)
            for key,value in regcoil_output.items():
                print('{} : {:e}'.format(key,value))
            np.testing.assert_almost_equal(regcoil_output['err_max_B'],err_max_B[index])
            np.testing.assert_almost_equal(regcoil_output['max_j'],max_j[index],decimal=0)
            np.testing.assert_almost_equal(regcoil_output['cost_B'],cost_B[index])
            np.testing.assert_almost_equal(regcoil_output['cost_J'],cost_J[index],decimal=-8)
if __name__ == '__main__':
    unittest.main()