from toroidal_surface import *
import vector_field
import bnorm
import logging
import configparser
from opt_einsum import contract
from scipy.constants import mu_0
#an example of Regcoil version in python
def cost_surface(config,S=None,Sp=None):

    #initilization of the parameters

    lamb = float(config['other']['lamb'])
    Np = int(config['geometry']['Np'])
    ntheta_plasma = int(config['geometry']['ntheta_plasma'])
    ntheta_coil   = int(config['geometry']['ntheta_coil'])
    nzeta_plasma = int(config['geometry']['nzeta_plasma'])
    nzeta_coil   = int(config['geometry']['nzeta_coil'])
    mpol_coil  = int(config['geometry']['mpol_coil'])
    ntor_coil  = int(config['geometry']['ntor_coil'])
    net_poloidal_current_Amperes = float(config['other']['net_poloidal_current_Amperes'])/Np#11884578.094260072
    net_toroidal_current_Amperes = float(config['other']['net_toroidal_current_Amperes'])#0
    curpol=float(config['other']['curpol'])#4.9782004309255496
    phisize=(mpol_coil,ntor_coil)
    path_plasma=str(config['geometry']['path_plasma'])#'code/li383/plasma_surf.txt'
    path_cws=str(config['geometry']['path_cws'])#'code/li383/cws.txt'
    path_bnorm=str(config['other']['path_bnorm'])#'code/li383/bnorm.txt'
    path_output=str(config['other']['path_output'])#'coeff_full_opt'
    dask = config['other']['dask']=='True'
    cupy = config['other']['cupy']=='True' # dask is needed to use cupy
    
    if dask :
        chunk_theta_coil=int(config['Dask_parameters']['chunk_theta_coil'])
        chunk_zeta_coil=int(config['Dask_parameters']['chunk_zeta_coil'])
        chunk_theta_plasma=int(config['Dask_parameters']['chunk_theta_plasma'])
        chunk_zeta_plasma=int(config['Dask_parameters']['chunk_zeta_plasma'])
        chunk_theta=int(config['Dask_parameters']['chunk_theta'])

    #initialization of the surfaces
    if S is None:
        S_parametrization=Toroidal_surface.load_file(path_cws)
        S=Toroidal_surface(S_parametrization,(ntheta_coil,nzeta_coil),Np)
    if Sp is None:
        Sp_parametrization=Toroidal_surface.load_file(path_plasma)
        Sp=Toroidal_surface(Sp_parametrization,(ntheta_plasma,nzeta_plasma),Np)

    

    #tensors computations
    rot_tensor=vector_field.get_rot_tensor(Np)
    T=vector_field.get_tensor_distance(S,Sp,rot_tensor)
    matrixd_phi=vector_field.get_matrix_dPhi(phisize,S.grids)
    Qj=vector_field.compute_Qj(matrixd_phi,S.dpsi,S.dS)
    if dask :
        import dask.array as da
        if cupy:
            import cupy as cp
            f= lambda x : x.map_blocks(cp.asarray,dtype=np.float64)
        else : f= lambda x : x
        dask_rot_tensor = f(da.from_array(rot_tensor, asarray=False))
        dask_matrixd_phi= f(da.from_array(matrixd_phi,chunks={1:chunk_theta_coil,2:chunk_zeta_coil}, asarray=False))
        dask_dpsi= f(da.from_array(S.dpsi,chunks=(2,3,chunk_theta_coil,chunk_zeta_coil), asarray=False))
        dask_normalp= f(da.from_array(Sp.n,chunks=(3,chunk_theta_plasma,chunk_zeta_plasma), asarray=False))
        dask_T= f(da.from_array(T,chunks=(3,chunk_theta_coil,chunk_zeta_coil,chunk_theta_plasma,chunk_zeta_plasma,3), asarray=False))
        dask_dS= f(da.from_array(S.dS,chunks=(chunk_theta_coil,chunk_zeta_coil), asarray=False))
        dask_D=1/(da.linalg.norm(dask_T,axis=-1)**3)
        dask_DD=1/(da.linalg.norm(dask_T,axis=-1)**5)
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        dask_eijk= f(da.from_array(eijk, asarray=False))
        LS_dask=contract('ijklmn,ojkw,ipz,wzjk,qnp,ijklm,qlm->olm',dask_T,dask_matrixd_phi,dask_rot_tensor,dask_dpsi,dask_eijk,dask_D,dask_normalp,optimize=True)
        LS=((mu_0/(4*np.pi))*LS_dask.compute()/(ntheta_coil*nzeta_coil))
        if cupy :
            LS=LS.get() #to obtain an np.array
    else:
        LS=vector_field.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
    array_bnorm=curpol*bnorm.get_bnorm(path_bnorm,Sp)
    ### Regcoil:
    cost_surface_output={}
    LS_matrix=np.transpose(np.reshape(LS[2:,:,:],(LS.shape[0]-2,-1)))#matrix shape
    BTn=net_poloidal_current_Amperes*LS[0]+net_toroidal_current_Amperes*LS[1]+array_bnorm
    BTn_flat=-BTn.flatten()# - for 2 reasons: we use inward convention (thus -array_bnorm)
    # and we want to eliminate the effect of the net currents
    

    #we take the adjoint, note that we have to take care of the different innerproduct
    plasma_dS_normalized=Sp.dS.flatten()/(Sp.nbpts[0]*Sp.nbpts[1])
    inside_M_lambda=np.einsum('ij,i,ik->jk',LS_matrix,plasma_dS_normalized,LS_matrix)\
        +lamb*Qj[2:,2:]
    RHS=(np.einsum('ij,i,i->j',LS_matrix,plasma_dS_normalized, BTn_flat)-lamb*(net_poloidal_current_Amperes*Qj[2:,0]+net_toroidal_current_Amperes*Qj[2:,1]) )
    j_S_partial= np.linalg.solve(inside_M_lambda,RHS)
    j_S=np.concatenate(([net_poloidal_current_Amperes,net_toroidal_current_Amperes],j_S_partial))
    
    # we save the results

    B_err= (LS_matrix @ j_S_partial)- BTn_flat
    cost_surface_output['err_max_B']=np.max(np.abs(B_err))
    cost_surface_output['max_j']=np.max(np.linalg.norm(np.einsum('oijk,kdij,ij,o->ijd',matrixd_phi,S.dpsi,1/S.dS,j_S,optimize=True),axis=2))
    cost_surface_output['cost_B']=Np*np.einsum('i,i,i->',B_err,B_err,plasma_dS_normalized)
    cost_surface_output['cost_J']=Np*np.einsum('i,ij,j->',j_S,Qj,j_S)
    return cost_surface_output


