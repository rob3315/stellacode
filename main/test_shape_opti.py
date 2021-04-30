import unittest
from regcoil import *
from shape_opti import *
import numpy as np
import logging

class Test_shape_opti(unittest.TestCase):
    @unittest.skip
    def test_LS_derivation_dim(self):
        phisize=6,7
        Np=4
        logging.basicConfig(level='DEBUG')
        surface_parametrization=Toroidal_surface.load_file('code/data/li383/cws.txt')
        ls=len(surface_parametrization[0])#total number of hamonics
        S=Toroidal_surface(surface_parametrization,(30,31),Np)
        Sp_parametrization=Toroidal_surface.load_file('code/data/li383/plasma_surf.txt')
        Sp=Toroidal_surface(Sp_parametrization,(32,33),Np)
        theta,dtildetheta,dtheta,dSdtheta = S.get_theta_pertubation()
        
        rot_tensor=vector_field.get_rot_tensor(Np)
    
        # we compute LS
        T=vector_field.get_tensor_distance(S,Sp,rot_tensor)
        matrixd_phi=vector_field.get_matrix_dPhi(phisize,S.grids)
        Qj=vector_field.compute_Qj(matrixd_phi,S.dpsi,S.dS)
        LS=vector_field.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
        dLS=vector_field.compute_dLS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n,theta,dtildetheta)
        dQj=vector_field.compute_dQjdtheta(matrixd_phi,S.dpsi,S.dS,dtildetheta,dSdtheta)
    @unittest.skip
    def test_several_derivation(self):
        #dL, dN
        lu,lv=32,32
        phisize=8,8
        Np=3
        eps=1e-8
        logging.basicConfig(level='DEBUG')
        surface_parametrization=Toroidal_surface.load_file('code/data/li383/cws.txt')
        ls=len(surface_parametrization[0])#total number of hamonics
        S=Toroidal_surface(surface_parametrization,(lu,lv),Np)
        Sp_parametrization=Toroidal_surface.load_file('code/data/li383/plasma_surf.txt')
        Sp=Toroidal_surface(Sp_parametrization,(lu,lv),Np)
        theta,dtildetheta,dtheta,dSdtheta=S.get_theta_pertubation()
        
        rot_tensor=vector_field.get_rot_tensor(Np)
    
        # we compute LS
        T=vector_field.get_tensor_distance(S,Sp,rot_tensor)
        matrixd_phi=vector_field.get_matrix_dPhi(phisize,S.grids)
        Qj=vector_field.compute_Qj(matrixd_phi,S.dpsi,S.dS)
        LS=vector_field.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)

        perturb=(2*np.random.random(2*ls)-1)
        new_param=Toroidal_surface.change_param(surface_parametrization, eps*perturb)
        new_S=Toroidal_surface(new_param,(lu,lv),3)

        new_T=vector_field.get_tensor_distance(new_S,Sp,rot_tensor)
        new_Qj=vector_field.compute_Qj(matrixd_phi,new_S.dpsi,new_S.dS)
        new_LS=vector_field.compute_LS(new_T,matrixd_phi,new_S.dpsi,rot_tensor,Sp.n)

        #test dLS
        #dLS=vector_field.compute_dLS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n,theta,dtildetheta)
        #d_LS_num=(new_LS-LS)/eps
        #d_LS=np.einsum('i,ijkl->jkl',perturb,dLS)
        #np.testing.assert_array_almost_equal(d_LS_num,d_LS)

        #test dpsi
        d_psi_num=(new_S.dpsi-S.dpsi)/eps
        d_psi=np.einsum('a,aijlk,wlij->wkij',perturb,dtildetheta,S.dpsi)
        d_psi2=np.einsum('a,aijlk->lkij',perturb,dtheta)
        np.testing.assert_array_almost_equal(d_psi2,d_psi_num,decimal=5)
        np.testing.assert_array_almost_equal(d_psi,d_psi2,decimal=5)
        #test dQS
        dQj=vector_field.compute_dQjdtheta(matrixd_phi,S.dpsi,S.dS,dtheta,dSdtheta)
        d_Qj_num=(new_Qj-Qj)/eps
        d_Qj=np.einsum('a,a...->...',perturb,dQj)
        np.testing.assert_array_almost_equal(d_Qj_num,d_Qj,decimal=0)
    def test_shape_gradient(self):
        #regcoil param
        lamb=1.2e-14
        ntheta_plasma = 16
        ntheta_coil   = 16
        nzeta_plasma = 16
        nzeta_coil   = 16
        mpol_coil  = 4
        ntor_coil  = 4
        Np=3 #symmetry
        net_poloidal_current_Amperes = 11884578.094260072/Np
        net_toroidal_current_Amperes = 0.#0.3#/(2*np.pi)
        curpol=4.9782004309255496# convention for the bnorm data, 
        path_bnorm='code/data/li383/bnorm.txt'
        path_cws='code/data/li383/cws.txt'
        path_plasma='code/data/li383/plasma_surf.txt'
        regcoil_param=Regcoil_param(lamb,Np,path_cws, path_plasma,path_bnorm,curpol,net_poloidal_current_Amperes,net_toroidal_current_Amperes, ntheta_plasma = ntheta_plasma,ntheta_coil = ntheta_coil,nzeta_plasma = nzeta_plasma, nzeta_coil = nzeta_coil, mpol_coil = mpol_coil, ntor_coil = ntor_coil)

        
        S_parametrization=Toroidal_surface.load_file(path_cws)
        S=Toroidal_surface(S_parametrization,(ntheta_coil,nzeta_coil),Np)
        eps=1e-6
        ls=len(S_parametrization[0])
        #we compute the shape derivative
        result1=get_shape_gradient(regcoil_param,S)
        
        #we apply one pertubation
        perturb=(2*np.random.random(2*ls)-1)
        new_param=Toroidal_surface.change_param(S_parametrization, eps*perturb)
        new_S=Toroidal_surface(new_param,(ntheta_coil,nzeta_coil),Np)
        result2=get_shape_gradient(regcoil_param,new_S)

        #we list the element to test
        quantities=['LS_matrix','RHS','j_S_partial','cost_J','tmp','cost_B','cost']
        gradient_quantities=['dLS_matrix_dtheta','dRHS_dtheta','dj_S_partial_dtheta','dcost_J_dtheta','grad_tmp','dcost_B_dtheta','shape_gradient']
        decimal_precision=[8,8,-4,-13,2,2,2]
        for quantity,grad,precision in zip(quantities,gradient_quantities,decimal_precision):
            print(quantity,grad)
            q1,grad_q1 = result1[quantity],result1[grad]
            q2= result2[quantity]

            dq_num=(q2-q1)/eps
            dq=np.einsum('a,a...->...',perturb,grad_q1)
            np.testing.assert_array_almost_equal(dq,dq_num,decimal=precision)
            print(dq.shape)


if __name__ == '__main__':
    unittest.main()