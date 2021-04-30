from toroidal_surface import *
import vector_field
import bnorm
import logging
from opt_einsum import contract
#an example of regcoil version in python
def get_shape_gradient(regcoil_param,S):
    result={}
    #same as regcoil param exept S which is given directly
    #initilization of the parameters
    lamb = regcoil_param.lamb
    Np = regcoil_param.Np
    path_plasma = regcoil_param.path_plasma
    path_bnorm = regcoil_param.path_bnorm
    net_poloidal_current_Amperes = regcoil_param.net_poloidal_current_Amperes
    net_toroidal_current_Amperes = regcoil_param.net_toroidal_current_Amperes
    ntheta_plasma = regcoil_param.ntheta_plasma
    nzeta_plasma = regcoil_param.nzeta_plasma
    phisize = regcoil_param.phisize
    curpol= regcoil_param.curpol

    #initialization of the plasma surface
    Sp_parametrization=Toroidal_surface.load_file(path_plasma)
    Sp=Toroidal_surface(Sp_parametrization,(ntheta_plasma,nzeta_plasma),Np)
    #standard regcoil computations
    rot_tensor=vector_field.get_rot_tensor(Np)
    T=vector_field.get_tensor_distance(S,Sp,rot_tensor)
    matrixd_phi=vector_field.get_matrix_dPhi(phisize,S.grids)
    Qj=vector_field.compute_Qj(matrixd_phi,S.dpsi,S.dS)
    array_bnorm=curpol*bnorm.get_bnorm(path_bnorm,Sp)
    LS=vector_field.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
    #shape derivation
    theta,dtildetheta,dtheta,dSdtheta=S.get_theta_pertubation()
    dLSdtheta=vector_field.compute_dLS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n,theta,dtildetheta)
    dQj=vector_field.compute_dQjdtheta(matrixd_phi,S.dpsi,S.dS,dtheta,dSdtheta)

    ### matrix versions:
    output={}
    LS_matrix=np.transpose(np.reshape(LS[2:,:,:],(LS.shape[0]-2,-1)))#matrix shape
    BTn=net_poloidal_current_Amperes*LS[0]+net_toroidal_current_Amperes*LS[1]+array_bnorm
    BTn_flat=-BTn.flatten()
    #shape derivative of those matrices
    dLS_matrix_dtheta=np.swapaxes(np.reshape(dLSdtheta[:,2:,:,:],(dLSdtheta.shape[0],dLSdtheta.shape[1]-2,-1)), 1, 2)
    dBTn_flat_dtheta=  np.reshape(net_poloidal_current_Amperes*dLSdtheta[:,0]+net_toroidal_current_Amperes*dLSdtheta[:,1],(dLSdtheta.shape[0],-1))
    
    result['LS_matrix']=LS_matrix
    result['dLS_matrix_dtheta']=dLS_matrix_dtheta
    #we take the adjoint, note that we have to take care of the different innerproduct
    plasma_dS_normalized=Sp.dS.flatten()/(Sp.nbpts[0]*Sp.nbpts[1])
    inside_M_lambda=np.einsum('ij,i,ik->jk',LS_matrix,plasma_dS_normalized,LS_matrix)\
        +lamb*Qj[2:,2:]
    
    dinside_M_lambda_dtheta=contract('oij,i,ik->ojk',dLS_matrix_dtheta,plasma_dS_normalized,LS_matrix)+contract('ij,i,oik->ojk',LS_matrix,plasma_dS_normalized,dLS_matrix_dtheta)\
        +lamb*dQj[:,2:,2:]
    
    RHS=(np.einsum('ij,i,i->j',LS_matrix,plasma_dS_normalized, BTn_flat)-lamb*(net_poloidal_current_Amperes*Qj[2:,0]+net_toroidal_current_Amperes*Qj[2:,1]) )
    dRHS_dtheta=np.einsum('oij,i,i->oj',dLS_matrix_dtheta,plasma_dS_normalized, BTn_flat)-np.einsum('ij,i,oi->oj',LS_matrix,plasma_dS_normalized, dBTn_flat_dtheta)-lamb*(net_poloidal_current_Amperes*dQj[:,2:,0]+net_toroidal_current_Amperes*dQj[:,2:,1])
    result['RHS']=RHS
    result['dRHS_dtheta']=dRHS_dtheta
    j_S_partial= np.linalg.solve(inside_M_lambda, RHS)
    j_S=np.concatenate(([net_poloidal_current_Amperes,net_toroidal_current_Amperes],j_S_partial))
    dj_S_partial_dtheta= np.linalg.solve(inside_M_lambda[np.newaxis,:,:], dRHS_dtheta)
    dj_S_partial_dtheta-=np.linalg.solve(inside_M_lambda[np.newaxis,:,:],np.einsum('oij,i->oj',dinside_M_lambda_dtheta, np.linalg.solve(inside_M_lambda,RHS)))
    result['j_S_partial']=j_S_partial
    result['dj_S_partial_dtheta']=dj_S_partial_dtheta
    #cost
    B_err= (LS_matrix @ j_S_partial)- BTn_flat
    result['cost_B']=Np*np.einsum('i,i,i->',B_err,B_err,plasma_dS_normalized)
    result['cost_J']=Np*np.einsum('i,ij,j->',j_S,Qj,j_S)
    result['cost']=result['cost_B']+lamb*result['cost_J']

    #computation of the gradient cost
    dcjdj= np.einsum('i,oij,j->o',j_S,dQj,j_S)
    dcjdj+= 2*np.einsum('i,ij,oj->o',j_S,Qj[:,2:],dj_S_partial_dtheta)
    result['dcost_J_dtheta']=Np*dcjdj

    result['tmp']=B_err
    tmp=np.einsum('oaij,a->oij',dLSdtheta,j_S)\
        +np.einsum('aij,oa->oij',LS[2:],dj_S_partial_dtheta)
    result['grad_tmp']=np.reshape(tmp,(tmp.shape[0],-1))
    dcbdj=(2*np.einsum('oi,i,i->o',np.reshape(tmp,(tmp.shape[0],-1)),B_err,plasma_dS_normalized))
    result['dcost_B_dtheta']=Np*dcbdj
    result['shape_gradient']=(dcbdj+lamb*dcjdj)*Np

    return result

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


