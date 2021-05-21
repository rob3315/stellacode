from toroidal_surface import *
import tools
import bnorm
import logging
import configparser
from opt_einsum import contract
from scipy.constants import mu_0
#an example of Regcoil version in python
def cost_surface(config,S=None,Sp=None):
    if config['other']['dask']=='True':
        return cost_surface_dask(config,S,Sp)
    else:
        return cost_surface_without_dask(config,S,Sp)
def cost_surface_without_dask(config,S,Sp):
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
        chunk_theta_coil=int(config['dask_parameters']['chunk_theta_coil'])
        chunk_zeta_coil=int(config['dask_parameters']['chunk_zeta_coil'])
        chunk_theta_plasma=int(config['dask_parameters']['chunk_theta_plasma'])
        chunk_zeta_plasma=int(config['dask_parameters']['chunk_zeta_plasma'])
        chunk_theta=int(config['dask_parameters']['chunk_theta'])

    #initialization of the surfaces
    if S is None:
        S_parametrization=Toroidal_surface.load_file(path_cws)
        S=Toroidal_surface(S_parametrization,(ntheta_coil,nzeta_coil),Np)
    if Sp is None:
        Sp_parametrization=Toroidal_surface.load_file(path_plasma)
        Sp=Toroidal_surface(Sp_parametrization,(ntheta_plasma,nzeta_plasma),Np)

    

    #tensors computations
    rot_tensor=tools.get_rot_tensor(Np)
    T=tools.get_tensor_distance(S,Sp,rot_tensor)
    matrixd_phi=tools.get_matrix_dPhi(phisize,S.grids)
    Qj=tools.compute_Qj(matrixd_phi,S.dpsi,S.dS)
    LS=tools.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
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

def cost_surface_dask(config,S,Sp):
    #new version with Lagrange multipliers
    import dask.array as da
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
    cupy = config['other']['cupy']=='True' # dask is needed to use cupy

    chunk_theta_coil=int(config['dask_parameters']['chunk_theta_coil'])
    chunk_zeta_coil=int(config['dask_parameters']['chunk_zeta_coil'])
    chunk_theta_plasma=int(config['dask_parameters']['chunk_theta_plasma'])
    chunk_zeta_plasma=int(config['dask_parameters']['chunk_zeta_plasma'])
    chunk_theta=int(config['dask_parameters']['chunk_theta'])

    #initialization of the surfaces
    if S is None:
        S_parametrization=Toroidal_surface.load_file(path_cws)
        S=Toroidal_surface(S_parametrization,(ntheta_coil,nzeta_coil),Np)
    if Sp is None:
        Sp_parametrization=Toroidal_surface.load_file(path_plasma)
        Sp=Toroidal_surface(Sp_parametrization,(ntheta_plasma,nzeta_plasma),Np)

    if cupy:
        import cupy as cp
        f= lambda x : x.map_blocks(cp.asarray,dtype=np.float64)
        f_np= lambda x : cp.asarray(x)
        get=lambda x : x.get()
    else : 
        f,f_np,get = lambda x : x,lambda x : x,lambda x : x
    #tensors computations
    rot_tensor=tools.get_rot_tensor(Np)
    T=tools.get_tensor_distance(S,Sp,rot_tensor)
    rot_tensor=f_np(rot_tensor)
    T=f(da.from_array(T,chunks=(3,chunk_theta_coil,chunk_zeta_coil,chunk_theta_plasma,chunk_zeta_plasma,3), asarray=False))
    matrixd_phi=f(da.from_array(tools.get_matrix_dPhi(phisize,S.grids),chunks={1:chunk_theta_coil,2:chunk_zeta_coil}, asarray=False))
    dpsi= f(da.from_array(S.dpsi,chunks=(2,3,chunk_theta_coil,chunk_zeta_coil), asarray=False))
    normalp= f(da.from_array(Sp.n,chunks=(3,chunk_theta_plasma,chunk_zeta_plasma), asarray=False))
    S_dS= f(da.from_array(S.dS,chunks=(chunk_theta_coil,chunk_zeta_coil), asarray=False))
    D=1/(da.linalg.norm(T,axis=-1)**3)
    DD=1/(da.linalg.norm(T,axis=-1)**5)
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    eijk= f_np(eijk)
    Qj=tools.compute_Qj(matrixd_phi,dpsi,S_dS)
    K=np.einsum('sijpqa,sijpq->sijpqa',T,D)
    LS=(mu_0/(4*np.pi))*contract('sijpqa,tijh,sbc,hcij,dab,dpq->tpq',K,matrixd_phi,rot_tensor,dpsi,eijk,normalp,optimize=True)/(ntheta_coil*nzeta_coil)

    Qj=get(Qj.compute())
    LS=get(LS.compute())
    BT=-curpol*bnorm.get_bnorm(path_bnorm,Sp)
    ### Regcoil:
    cost_surface_output={}
    #WARNING : LS contained the poloidal and toroidal currents coordinates
    Qj_inv=np.linalg.inv(Qj)
    LS_dagger=np.einsum('ut,tij,ij->uij',Qj_inv,LS,Sp.dS/Sp.npts)
    inside_M_lambda= lamb*np.eye(len(Qj[0]))+np.einsum('tpq,upq->tu',LS_dagger,LS)
    M_lambda=np.linalg.inv(inside_M_lambda)
    
    # we compute the 2 Lagrange multipliers
    LS_dagger_B_T=np.einsum('hpq,pq->h',LS_dagger,BT)
    m0,m1=np.linalg.inv(M_lambda[:2]@Qj_inv[:,:2]) @((M_lambda@LS_dagger_B_T)[:2] - [net_poloidal_current_Amperes,net_toroidal_current_Amperes])
    RHS=LS_dagger_B_T -[m0,m1]@Qj_inv[:2,:]
    j_S= M_lambda@RHS
    
    # we save the results

    B_err= np.einsum('hpq,h',LS,j_S)- BT
    cost_surface_output['err_max_B']=np.max(np.abs(B_err))
    cost_surface_output['max_j']=np.max(np.linalg.norm(np.einsum('oijk,kdij,ij,o->ijd',get(matrixd_phi.compute()),S.dpsi,1/S.dS,j_S,optimize=True),axis=2))
    cost_surface_output['cost_B']=Np*np.einsum('pq,pq,pq->',B_err,B_err,Sp.dS/Sp.npts)
    cost_surface_output['cost_J']=Np*np.einsum('i,ij,j->',j_S,Qj,j_S)
    return cost_surface_output
def cost_surface_dask_old(config,S,Sp):
    import dask.array as da
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
    cupy = config['other']['cupy']=='True' # dask is needed to use cupy

    chunk_theta_coil=int(config['dask_parameters']['chunk_theta_coil'])
    chunk_zeta_coil=int(config['dask_parameters']['chunk_zeta_coil'])
    chunk_theta_plasma=int(config['dask_parameters']['chunk_theta_plasma'])
    chunk_zeta_plasma=int(config['dask_parameters']['chunk_zeta_plasma'])
    chunk_theta=int(config['dask_parameters']['chunk_theta'])

    #initialization of the surfaces
    if S is None:
        S_parametrization=Toroidal_surface.load_file(path_cws)
        S=Toroidal_surface(S_parametrization,(ntheta_coil,nzeta_coil),Np)
    if Sp is None:
        Sp_parametrization=Toroidal_surface.load_file(path_plasma)
        Sp=Toroidal_surface(Sp_parametrization,(ntheta_plasma,nzeta_plasma),Np)

    if cupy:
        import cupy as cp
        f= lambda x : x.map_blocks(cp.asarray,dtype=np.float64)
        f_np= lambda x : cp.asarray(x)
        get=lambda x : x.get()
    else : 
        f,f_np,get = lambda x : x,lambda x : x,lambda x : x
    #tensors computations
    rot_tensor=tools.get_rot_tensor(Np)
    T=tools.get_tensor_distance(S,Sp,rot_tensor)
    rot_tensor=f_np(rot_tensor)
    T=f(da.from_array(T,chunks=(3,chunk_theta_coil,chunk_zeta_coil,chunk_theta_plasma,chunk_zeta_plasma,3), asarray=False))
    matrixd_phi=f(da.from_array(tools.get_matrix_dPhi(phisize,S.grids),chunks={1:chunk_theta_coil,2:chunk_zeta_coil}, asarray=False))
    dpsi= f(da.from_array(S.dpsi,chunks=(2,3,chunk_theta_coil,chunk_zeta_coil), asarray=False))
    normalp= f(da.from_array(Sp.n,chunks=(3,chunk_theta_plasma,chunk_zeta_plasma), asarray=False))
    S_dS= f(da.from_array(S.dS,chunks=(chunk_theta_coil,chunk_zeta_coil), asarray=False))
    D=1/(da.linalg.norm(T,axis=-1)**3)
    DD=1/(da.linalg.norm(T,axis=-1)**5)
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    eijk= f_np(eijk)
    Qj=tools.compute_Qj(matrixd_phi,dpsi,S_dS)
    K=np.einsum('sijpqa,sijpq->sijpqa',T,D)
    LS=(mu_0/(4*np.pi))*contract('sijpqa,tijh,sbc,hcij,dab,dpq->tpq',K,matrixd_phi,rot_tensor,dpsi,eijk,normalp,optimize=True)/(ntheta_coil*nzeta_coil)

    Qj=get(Qj.compute())
    LS=get(LS.compute())
   #if cupy :
   #    LS=LS.get() #to obtain an np.array
   #else:
   #    LS=tools.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
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
    cost_surface_output['max_j']=np.max(np.linalg.norm(np.einsum('oijk,kdij,ij,o->ijd',get(matrixd_phi.compute()),S.dpsi,1/S.dS,j_S,optimize=True),axis=2))
    cost_surface_output['cost_B']=Np*np.einsum('i,i,i->',B_err,B_err,plasma_dS_normalized)
    cost_surface_output['cost_J']=Np*np.einsum('i,ij,j->',j_S,Qj,j_S)
    return cost_surface_output

