from toroidal_surface import *
import vector_field
import bnorm
import logging
#an example of regcoil version in python
def regcoil(regcoil_param):

    #initilization of the parameters
    lamb = regcoil_param.lamb
    Np = regcoil_param.Np
    path_cws = regcoil_param.path_cws
    path_plasma = regcoil_param.path_plasma
    path_bnorm = regcoil_param.path_bnorm
    net_poloidal_current_Amperes = regcoil_param.net_poloidal_current_Amperes
    net_toroidal_current_Amperes = regcoil_param.net_toroidal_current_Amperes
    ntheta_plasma = regcoil_param.ntheta_plasma
    ntheta_coil = regcoil_param.ntheta_coil
    nzeta_plasma = regcoil_param.nzeta_plasma
    nzeta_coil = regcoil_param.nzeta_coil
    phisize = regcoil_param.phisize
    curpol= regcoil_param.curpol

    #initialization of the surfaces
    S_parametrization=Toroidal_surface.load_file(path_cws)
    S=Toroidal_surface(S_parametrization,(ntheta_coil,nzeta_coil),Np)

    Sp_parametrization=Toroidal_surface.load_file(path_plasma)
    Sp=Toroidal_surface(Sp_parametrization,(ntheta_plasma,nzeta_plasma),Np)

    #tensors computations
    rot_tensor=vector_field.get_rot_tensor(Np)
    T=vector_field.get_tensor_distance(S,Sp,rot_tensor)
    matrixd_phi=vector_field.get_matrix_dPhi(phisize,S.grids)
    Qj=vector_field.compute_Qj(matrixd_phi,S.dpsi,S.dS)
    
    #boldpsi=S.get_boldpsi()
    #j=vector_field.compute_j(boldpsi,matrixd_phi,Np)
    #Qj_old=vector_field.compute_Qj_old(j[0],S.dS)
    
    #dpsi_full=vector_field.compute_dpsi_full(S.dpsi,Np)# with rotation
    LS=vector_field.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
    logging.debug('LS shape : {}'.format(LS.shape))
    array_bnorm=curpol*bnorm.get_bnorm(path_bnorm,Sp)
    logging.debug('bnorm shape : {}'.format(array_bnorm.shape))

    ### Regcoil:
    regcoil_output={}
    LS_matrix=np.transpose(np.reshape(LS[2:,:,:],(LS.shape[0]-2,-1)))#matrix shape
    #LS_dagger_matrix
    BTn=net_poloidal_current_Amperes*LS[0]+net_toroidal_current_Amperes*LS[1]+array_bnorm
    BTn_flat=-BTn.flatten()# - for 2 reasons: we use inward convention (thus -array_bnorm)
    # and we want to eliminate the effect of the net currents
    

    #we take the adjoint, note that we have to take care of the different innerproduct
    plasma_dS_normalized=Sp.dS.flatten()/(Sp.nbpts[0]*Sp.nbpts[1])
    inside_M_lambda=np.einsum('ij,i,ik->jk',LS_matrix,plasma_dS_normalized,LS_matrix)\
        +lamb*Qj[2:,2:]
    #M_lambda=np.linalg.inv(inside_M_lambda)# TODO : avoid inversion
    RHS=(np.einsum('ij,i,i->j',LS_matrix,plasma_dS_normalized, BTn_flat)-lamb*(net_poloidal_current_Amperes*Qj[2:,0]+net_toroidal_current_Amperes*Qj[2:,1]) )
    j_S_partial= np.linalg.solve(inside_M_lambda,RHS)
    j_S=np.concatenate(([net_poloidal_current_Amperes,net_toroidal_current_Amperes],j_S_partial))
    logging.debug('j_S shape : {}'.format(j_S.shape))
    
    # we save the results

    B_err= (LS_matrix @ j_S_partial)- BTn_flat
    regcoil_output['err_max_B']=np.max(np.abs(B_err))
    regcoil_output['max_j']=np.max(np.linalg.norm(np.einsum('oijk,kdij,ij,o->ijd',matrixd_phi,S.dpsi,1/S.dS,j_S,optimize=True),axis=2))
    regcoil_output['cost_B']=Np*np.einsum('i,i,i->',B_err,B_err,plasma_dS_normalized)
    regcoil_output['cost_J']=Np*np.einsum('i,ij,j->',j_S,Qj,j_S)
    return regcoil_output

class Regcoil_param():
    def __init__(self,lamb,Np,path_cws, path_plasma,path_bnorm,curpol,net_poloidal_current_Amperes,net_toroidal_current_Amperes, ntheta_plasma = 64,ntheta_coil = 64,nzeta_plasma = 64, nzeta_coil = 64, mpol_coil = 8, ntor_coil = 8):
        self.lamb=lamb
        self.Np=Np
        self.path_cws=path_cws
        self.path_plasma=path_plasma
        self.path_bnorm=path_bnorm
        self.curpol=curpol
        self.net_poloidal_current_Amperes=net_poloidal_current_Amperes
        self.net_toroidal_current_Amperes=net_toroidal_current_Amperes
        self.ntheta_plasma=ntheta_plasma
        self.ntheta_coil=ntheta_coil
        self.nzeta_plasma=nzeta_plasma
        self.nzeta_coil=nzeta_coil
        self.phisize=(mpol_coil,ntor_coil)


