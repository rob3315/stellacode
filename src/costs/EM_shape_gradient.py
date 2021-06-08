import logging
import numpy as np
from opt_einsum import contract
import configparser
from scipy.constants import mu_0
import dask.array as da
import dask

from src.surface.surface_Fourier import Surface_Fourier
from src.costs.abstract_shape_gradient import Abstract_shape_gradient
from src.costs.EM_cost import EM_cost_dask
import src.tools as tools
import src.tools.bnorm as bnorm
class EM_shape_gradient(Abstract_shape_gradient):
    def __init__(self,path_config_file=None,config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config=config
        self.Np=int(config['geometry']['Np'])
        ntheta_plasma = int(config['geometry']['ntheta_plasma'])
        self.ntheta_coil   = int(config['geometry']['ntheta_coil'])
        nzeta_plasma = int(config['geometry']['nzeta_plasma'])
        self.nzeta_coil   = int(config['geometry']['nzeta_coil'])
        self.net_poloidal_current_Amperes = float(config['other']['net_poloidal_current_Amperes'])/self.Np#11884578.094260072
        self.net_toroidal_current_Amperes = float(config['other']['net_toroidal_current_Amperes'])#0
        curpol=float(config['other']['curpol'])#4.9782004309255496

        self.lamb=float(config['other']['lamb'])

        path_plasma=str(config['geometry']['path_plasma'])#'code/li383/plasma_surf.txt'
        path_cws=str(config['geometry']['path_cws'])#'code/li383/cws.txt'
        path_bnorm=str(config['other']['path_bnorm'])#'code/li383/bnorm.txt'
        self.path_output=str(config['other']['path_output'])#'coeff_full_opt'

        mpol_coil  = int(config['geometry']['mpol_coil'])
        ntor_coil  = int(config['geometry']['ntor_coil'])
        phisize=(mpol_coil,ntor_coil)

        if config['other']['dask']!= 'True':
            raise Exception('dask is needed for Shape gradient')
        self.chunk_theta_coil=int(config['dask_parameters']['chunk_theta_coil'])
        self.chunk_zeta_coil=int(config['dask_parameters']['chunk_zeta_coil'])
        self.chunk_theta_plasma=int(config['dask_parameters']['chunk_theta_plasma'])
        self.chunk_zeta_plasma=int(config['dask_parameters']['chunk_zeta_plasma'])
        self.chunk_theta=int(config['dask_parameters']['chunk_theta'])
        
        #initialization of the surfaces
        self.S_parametrization=Surface_Fourier.load_file(path_cws)
        S=Surface_Fourier(self.S_parametrization,(self.ntheta_coil,self.nzeta_coil),self.Np)
        Sp_parametrization=Surface_Fourier.load_file(path_plasma)
        self.Sp=Surface_Fourier(Sp_parametrization,(ntheta_plasma,nzeta_plasma),self.Np)
        self.rot_tensor=tools.get_rot_tensor(self.Np)
        self.matrixd_phi=tools.get_matrix_dPhi(phisize,S.grids)
        self.array_bnorm=curpol*bnorm.get_bnorm(path_bnorm,self.Sp)

        self.dask_rot_tensor = da.from_array(self.rot_tensor, asarray=False)
        self.dask_matrixd_phi=da.from_array(self.matrixd_phi,chunks={1:self.chunk_theta_coil,2:self.chunk_zeta_coil}, asarray=False)
    def cost(self,S):
        EM_cost_dic=EM_cost_dask(self.config,S=S,Sp=self.Sp)
        EM_cost=EM_cost_dic['cost_B']+ self.lamb*EM_cost_dic['cost_J']
        logging.info('Chi_B : {:5e}, Chi_j = {:5e}, EM cost {:5e}'.format(EM_cost_dic['cost_B'],EM_cost_dic['cost_J'],EM_cost))
        return EM_cost
    def shape_gradient(self,S,theta_pertubation):
        theta,dtildetheta,dtheta,dSdtheta=theta_pertubation
        result=self.compute_gradient_of(S=S)
        I_vector,I_matrix=result['I1']
        EM_grad=self.Np*(np.einsum('ija,oija,ij->o',I_vector,theta,S.dS/S.npts)+np.einsum('ijab,oijab,ij->o',I_matrix,dtildetheta,S.dS/S.npts))
        return EM_grad

    def compute_gradient_of(self,paramS=None,S=None):
        #compute the shape gradient by a optimization first method
        result={}
        if paramS is None:
            if S is None:
                raise Exception('No surface given')
        else :
            S=Surface_Fourier(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        # for latter, when we will implement GPU support
        f,f_np,get = lambda x : x,lambda x : x,lambda x : x
        #tensors computations
        T=tools.get_tensor_distance(S,self.Sp,self.rot_tensor)
        result['T']=T
        T=f(da.from_array(T,chunks=(3,self.chunk_theta_coil,self.chunk_zeta_coil,self.chunk_theta_plasma,self.chunk_zeta_plasma,3), asarray=False))
        matrixd_phi=f(da.from_array(self.matrixd_phi,chunks={1:self.chunk_theta_coil,2:self.chunk_zeta_coil}, asarray=False))
        dpsi= f(da.from_array(S.dpsi,chunks=(2,3,self.chunk_theta_coil,self.chunk_zeta_coil), asarray=False))
        normalp= f(da.from_array(self.Sp.n,chunks=(3,self.chunk_theta_plasma,self.chunk_zeta_plasma), asarray=False))
        S_dS= f(da.from_array(S.dS,chunks=(self.chunk_theta_coil,self.chunk_zeta_coil), asarray=False))
        D=1/(da.linalg.norm(T,axis=-1)**3)
        DD=1/(da.linalg.norm(T,axis=-1)**5)
        Qj=tools.compute_Qj(matrixd_phi,dpsi,S_dS)
        LS=(mu_0/(4*np.pi))*contract('ijklmn,ojkw,ipz,wzjk,qnp,ijklm,qlm->olm',T,matrixd_phi,self.rot_tensor,dpsi,tools.eijk,D,normalp,optimize=True)/(S.nbpts[0]*S.nbpts[1])

        # We solve the inverse problem
        BT=-self.array_bnorm
        LS_R=LS[2:]
        Qj_inv_R=np.linalg.inv(Qj[2:,2:])
        LS_dagger_R=np.einsum('ut,tij,ij->uij',Qj_inv_R,LS_R,self.Sp.dS/self.Sp.npts)
        inside_M_lambda_R= self.lamb*np.eye(LS_R.shape[0])+np.einsum('tpq,upq->tu',LS_dagger_R,LS_R)
        M_lambda_R=np.linalg.inv(inside_M_lambda_R)
        
        # we compute the full Right Hand Side
        B_tilde=BT-np.einsum('tpq,t',LS[:2],[self.net_poloidal_current_Amperes ,self.net_toroidal_current_Amperes])
        LS_dagger_B_tilde=np.einsum('hpq,pq->h',LS_dagger_R,B_tilde)
        RHS=LS_dagger_B_tilde-self.lamb*Qj_inv_R@Qj[2:,:2]@[self.net_poloidal_current_Amperes ,self.net_toroidal_current_Amperes]
        j_S_R= M_lambda_R@RHS
        j_S=np.concatenate(([self.net_poloidal_current_Amperes,self.net_toroidal_current_Amperes],j_S_R))
        
        # we save the results

        B_err= np.einsum('hpq,h',LS,j_S)- BT
        # we save the results
        j_space_to_vector=contract('oijk,klij,ij->oijl',matrixd_phi,dpsi,1/S.dS)
        def j_to_vector(j):
            """from the space of divergence free vector field to 3d vector field on T"""
            if len(j)==len(j_space_to_vector):
                return np.einsum('o,oijl->ijl',j,j_space_to_vector)
            elif len(j)==len(j_space_to_vector)-2:
                return np.einsum('o,oijl->ijl',j,j_space_to_vector[2:])
            else:
                 raise Exception('dimension error in j_to_vector')
        B_err=np.einsum('opq,o->pq',LS,j_S)-BT
        j_S_vector=contract('o,oijl->ijl',j_S,j_space_to_vector)

        #We start to compute the shape gradient
        
        K=np.einsum('sijpql,sijpq->sijpql',T,D)
        DxK= -(np.einsum('sijpq,sab->sijpqab',D,self.rot_tensor)-3*np.einsum('sijpq,sijpqa,sijpqb,sbc->sijpqac',DD,T,T,self.rot_tensor))
        Zp_aux= -(mu_0/(4*np.pi))*contract('sijpqa,sbe,abd,dpq,pq->ijpqe',K,self.rot_tensor,tools.eijk,normalp,self.Sp.dS/self.Sp.npts,optimize=True)
        def Zp(k,j):
            return contract('ijpqe,pq,ija->ijae',Zp_aux,k,j)
        def Z_p_hat(k,j):
            return (mu_0/(4*np.pi))*contract('saf,sijpqad,sce,pq,cpq,ebd,ijb,pq->ijf',self.dask_rot_tensor,DxK,self.dask_rot_tensor,k,normalp,tools.eijk,j,self.Sp.dS/self.Sp.npts)
        def dLdtheta(k,j):
            return (-1*Z_p_hat(k,j),-1*Zp(k,j))
        def dQdtheta(j1,j2):
            return (contract('ijl,ijk->ijlk',j1,j2)+contract('ijl,ijk->ijlk',j2,j1)+np.einsum('ijk,ijk,ijab->ijab',j1,j2,-np.eye(3)+np.einsum('aij,bij->ijab',S.n,S.n)))
        I1_vector,I1_matrix =dLdtheta(2*B_err,j_S_vector)
        I1_matrix2=self.lamb*dQdtheta(j_S_vector,j_S_vector)
        x, y = dask.optimize(I1_vector, I1_matrix+I1_matrix2)
        result['I1']=da.compute(x,y)
        return result
    def compute_gradient_df(self,paramS):
        #compute the shape gradient by a differentiation first method
        result={}
        S=Surface_Fourier(paramS,(self.ntheta_coil,self.nzeta_coil),self.Np)
        theta,dtildetheta,dtheta,dSdtheta=S.get_theta_pertubation()
        T=tools.get_tensor_distance(S,self.Sp,self.rot_tensor)
        #LS=tools.compute_LS(T,self.matrixd_phi,S.dpsi,self.rot_tensor,self.Sp.n)
        Qj=tools.compute_Qj(self.matrixd_phi,S.dpsi,S.dS)
        #shape derivation
        dask_theta=da.from_array(theta,chunks=(self.chunk_theta,self.chunk_theta_coil,self.chunk_zeta_coil,3), asarray=False)
        dask_dpsi=da.from_array(S.dpsi,chunks=(2,3,self.chunk_theta_coil,self.chunk_zeta_coil), asarray=False)
        
        dask_normalp=da.from_array(self.Sp.n,chunks=(3,self.chunk_theta_plasma,self.chunk_zeta_plasma), asarray=False)
        dask_T=da.from_array(T,chunks=(3,self.chunk_theta_coil,self.chunk_zeta_coil,self.chunk_theta_plasma,self.chunk_zeta_plasma,3), asarray=False)
        dask_dtildetheta=da.from_array(dtildetheta,chunks=(self.chunk_theta,self.chunk_theta_coil,self.chunk_zeta_coil,3,3), asarray=False)
        dask_dtheta=da.from_array(dtheta,chunks=(self.chunk_theta,self.chunk_theta_coil,self.chunk_zeta_coil,2,3), asarray=False)
        dask_dSdtheta=da.from_array(dSdtheta,chunks=(self.chunk_theta,self.chunk_theta_coil,self.chunk_zeta_coil), asarray=False)
        dask_dS=da.from_array(S.dS,chunks=(self.chunk_theta_coil,self.chunk_zeta_coil), asarray=False)
        
        #dask_D=1/(da.linalg.norm(dask_T,axis=-1)**3)
        #dask_DD=1/(da.linalg.norm(dask_T,axis=-1)**5)
        D=1/(np.linalg.norm(T,axis=-1)**3)
        DD=1/(np.linalg.norm(T,axis=-1)**5)
        dask_D=da.from_array(D,chunks=(3,self.chunk_theta_coil,self.chunk_zeta_coil,self.chunk_theta_plasma,self.chunk_zeta_plasma), asarray=False)
        dask_DD=da.from_array(DD,chunks=(3,self.chunk_theta_coil,self.chunk_zeta_coil,self.chunk_theta_plasma,self.chunk_zeta_plasma), asarray=False)
        LS_dask=contract('ijklmn,ojkw,ipz,wzjk,qnp,ijklm,qlm->olm',dask_T,self.dask_matrixd_phi,self.dask_rot_tensor,dask_dpsi,tools.eijk,dask_D,dask_normalp,optimize=True)
        LS=(mu_0/(4*np.pi))*LS_dask.compute()/(self.ntheta_coil*self.nzeta_coil)

        dLS_dask=-contract('inb,ajkb,ojkw,ipz,wzjk,qnp,ijklm,qlm->aolm',self.dask_rot_tensor,dask_theta,self.dask_matrixd_phi,self.dask_rot_tensor,dask_dpsi,tools.eijk,dask_D,dask_normalp,optimize=True)
        dLS_dask+=3*contract('ibc,ajkc,ijklmb,ijklmn,ojkw,ipz,wzjk,qnp,ijklm,qlm->aolm',self.dask_rot_tensor,dask_theta,dask_T,dask_T,self.dask_matrixd_phi,self.dask_rot_tensor,dask_dpsi,tools.eijk,dask_DD,dask_normalp,optimize=True)
        dLS_dask+=contract('ijklmn,ojkw,ipz,ajkbz,wbjk,qnp,ijklm,qlm->aolm',dask_T,self.dask_matrixd_phi,self.dask_rot_tensor,dask_dtildetheta,dask_dpsi,tools.eijk,dask_D,dask_normalp,optimize=True)
        dLSdtheta=(mu_0/(4*np.pi))*dLS_dask.compute()/(self.ntheta_coil*self.nzeta_coil)
        dask_dQj=tools.compute_dQjdtheta(self.dask_matrixd_phi,dask_dpsi,dask_dS,dask_dtheta,dask_dSdtheta)
        dQj=dask_dQj.compute()
        result['Qj']=Qj
        result['dQj']=dQj
        result['dLSdtheta']=dLSdtheta
        result['LS']=LS

        ### matrix versions:
        output={}
        LS_matrix=np.transpose(np.reshape(LS[2:,:,:],(LS.shape[0]-2,-1)))#matrix shape
        BTn=self.net_poloidal_current_Amperes*LS[0]+self.net_toroidal_current_Amperes*LS[1]+self.array_bnorm
        BTn_flat=-BTn.flatten()
        result['array_bnorm']=self.array_bnorm
        #shape derivative of those matrices
        dLS_matrix_dtheta=np.swapaxes(np.reshape(dLSdtheta[:,2:,:,:],(dLSdtheta.shape[0],dLSdtheta.shape[1]-2,-1)), 1, 2)
        dLS_matrix_dtheta_dask=da.from_array(dLS_matrix_dtheta,chunks={0:1})#for parallelisation
        dBTn_flat_dtheta= - np.reshape(self.net_poloidal_current_Amperes*dLSdtheta[:,0]+self.net_toroidal_current_Amperes*dLSdtheta[:,1],(dLSdtheta.shape[0],-1))
        
        result['LS_matrix']=LS_matrix
        result['dLS_matrix_dtheta']=dLS_matrix_dtheta

        #we take the adjoint, note that we have to take care of the different innerproduct
        plasma_dS_normalized=self.Sp.dS.flatten()/(self.Sp.nbpts[0]*self.Sp.nbpts[1])
        inside_M_lambda=contract('ij,i,ik->jk',LS_matrix,plasma_dS_normalized,LS_matrix)\
            +self.lamb*Qj[2:,2:]
        
        dinside_M_lambda_dtheta_dask=contract('oij,i,ik->ojk',dLS_matrix_dtheta_dask,plasma_dS_normalized,LS_matrix)+contract('ij,i,oik->ojk',LS_matrix,plasma_dS_normalized,dLS_matrix_dtheta_dask)
        dinside_M_lambda_dtheta=dinside_M_lambda_dtheta_dask.compute()+self.lamb*dQj[:,2:,2:]
        
        RHS=(contract('ij,i,i->j',LS_matrix,plasma_dS_normalized, BTn_flat)-self.lamb*(self.net_poloidal_current_Amperes*Qj[2:,0]+self.net_toroidal_current_Amperes*Qj[2:,1]) )
        dRHS_dtheta=contract('oij,i,i->oj',dLS_matrix_dtheta,plasma_dS_normalized, BTn_flat)+contract('ij,i,oi->oj',LS_matrix,plasma_dS_normalized, dBTn_flat_dtheta)-self.lamb*(self.net_poloidal_current_Amperes*dQj[:,2:,0]+self.net_toroidal_current_Amperes*dQj[:,2:,1])
        result['RHS']=RHS
        result['dRHS_dtheta']=dRHS_dtheta
        j_S_partial= np.linalg.solve(inside_M_lambda, RHS)
        j_S=np.concatenate(([self.net_poloidal_current_Amperes,self.net_toroidal_current_Amperes],j_S_partial))
        dj_S_partial_dtheta= np.linalg.solve(inside_M_lambda[np.newaxis,:,:], dRHS_dtheta)
        dj_S_partial_dtheta-=np.linalg.solve(inside_M_lambda[np.newaxis,:,:],contract('oij,i->oj',dinside_M_lambda_dtheta, np.linalg.solve(inside_M_lambda,RHS)))
        result['j_S_partial']=j_S_partial
        result['dj_S_partial_dtheta']=dj_S_partial_dtheta
        #cost
        B_err= (LS_matrix @ j_S_partial)- BTn_flat

        result['cost_B']=self.Np*contract('i,i,i->',B_err,B_err,plasma_dS_normalized)
        result['cost_J']=self.Np*contract('i,ij,j->',j_S,Qj,j_S)
        result['cost']=result['cost_B']+self.lamb*result['cost_J']
        dcjdj= np.einsum('i,oij,j->o',j_S,dQj,j_S)
        dcjdj+= 2*np.einsum('i,ij,oj->o',j_S,Qj[:,2:],dj_S_partial_dtheta)
        result['dcost_J_dtheta']=self.Np*dcjdj

        tmp=np.einsum('oaij,a->oij',dLSdtheta,j_S)\
            +np.einsum('aij,oa->oij',LS[2:],dj_S_partial_dtheta)
        
        dcbdj=(2*np.einsum('oi,i,i->o',np.reshape(tmp,(tmp.shape[0],-1)),B_err,plasma_dS_normalized))
        result['dcost_B_dtheta']=self.Np*dcbdj
        result['EM_shape_gradient']=(dcbdj+self.lamb*dcjdj)*self.Np

        return result