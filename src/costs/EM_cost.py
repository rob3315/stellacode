"""Various implementation of the main cost"""

import logging
import configparser
import numpy as np
from opt_einsum import contract
from scipy.constants import mu_0

from src.surface.surface_Fourier import Surface_Fourier
import src.tools as tools
import src.tools.bnorm as bnorm
# an example of Regcoil version in python


def EM_cost(config, S=None, Sp=None):
    """dispatch depending on the dask option in config"""
    if config['other']['dask'] == 'True':
        return EM_cost_dask(config, S, Sp)
    else:
        return EM_cost_without_dask(config, S, Sp)


def EM_cost_without_dask(config, S, Sp):
    # initilization of the parameters
    lamb = float(config['other']['lamb'])
    Np = int(config['geometry']['Np'])
    ntheta_plasma = int(config['geometry']['ntheta_plasma'])
    ntheta_coil = int(config['geometry']['ntheta_coil'])
    nzeta_plasma = int(config['geometry']['nzeta_plasma'])
    nzeta_coil = int(config['geometry']['nzeta_coil'])
    mpol_coil = int(config['geometry']['mpol_coil'])
    ntor_coil = int(config['geometry']['ntor_coil'])
    net_poloidal_current_Amperes = float(
        config['other']['net_poloidal_current_Amperes'])/Np  # 11884578.094260072
    net_toroidal_current_Amperes = float(
        config['other']['net_toroidal_current_Amperes'])  # 0
    curpol = float(config['other']['curpol'])  # 4.9782004309255496
    phisize = (mpol_coil, ntor_coil)
    # 'code/li383/plasma_surf.txt'
    path_plasma = str(config['geometry']['path_plasma'])
    path_cws = str(config['geometry']['path_cws'])  # 'code/li383/cws.txt'
    path_bnorm = str(config['other']['path_bnorm'])  # 'code/li383/bnorm.txt'
    path_output = str(config['other']['path_output'])  # 'coeff_full_opt'
    dask = config['other']['dask'] == 'True'
    cupy = config['other']['cupy'] == 'True'  # dask is needed to use cupy

    if dask:
        chunk_theta_coil = int(config['dask_parameters']['chunk_theta_coil'])
        chunk_zeta_coil = int(config['dask_parameters']['chunk_zeta_coil'])
        chunk_theta_plasma = int(
            config['dask_parameters']['chunk_theta_plasma'])
        chunk_zeta_plasma = int(config['dask_parameters']['chunk_zeta_plasma'])
        chunk_theta = int(config['dask_parameters']['chunk_theta'])

    # initialization of the surfaces
    if S is None:
        S_parametrization = Surface_Fourier.load_file(path_cws)
        S = Surface_Fourier(S_parametrization, (ntheta_coil, nzeta_coil), Np)
    if Sp is None:
        Sp_parametrization = Surface_Fourier.load_file(path_plasma)
        Sp = Surface_Fourier(Sp_parametrization,
                             (ntheta_plasma, nzeta_plasma), Np)

    # tensors computations
    rot_tensor = tools.get_rot_tensor(Np)
    T = tools.get_tensor_distance(S, Sp, rot_tensor)
    matrixd_phi = tools.get_matrix_dPhi(phisize, S.grids)
    Qj = tools.compute_Qj(matrixd_phi, S.dpsi, S.dS)
    LS = tools.compute_LS(T, matrixd_phi, S.dpsi, rot_tensor, Sp.n)
    array_bnorm = curpol*bnorm.get_bnorm(path_bnorm, Sp)
    # Regcoil:
    EM_cost_output = {}
    LS_matrix = np.transpose(np.reshape(
        LS[2:, :, :], (LS.shape[0]-2, -1)))  # matrix shape
    BTn = net_poloidal_current_Amperes * \
        LS[0]+net_toroidal_current_Amperes*LS[1]+array_bnorm
    # - for 2 reasons: we use inward convention (thus -array_bnorm)
    BTn_flat = -BTn.flatten()
    # and we want to eliminate the effect of the net currents

    # we take the adjoint, note that we have to take care of the different innerproduct
    plasma_dS_normalized = Sp.dS.flatten()/(Sp.nbpts[0]*Sp.nbpts[1])
    inside_M_lambda = np.einsum('ij,i,ik->jk', LS_matrix, plasma_dS_normalized, LS_matrix)\
        + lamb*Qj[2:, 2:]
    RHS = (np.einsum('ij,i,i->j', LS_matrix, plasma_dS_normalized, BTn_flat)-lamb *
           (net_poloidal_current_Amperes*Qj[2:, 0]+net_toroidal_current_Amperes*Qj[2:, 1]))
    j_S_partial = np.linalg.solve(inside_M_lambda, RHS)
    j_S = np.concatenate(
        ([net_poloidal_current_Amperes, net_toroidal_current_Amperes], j_S_partial))
    # we save the results
    B_err = (LS_matrix @ j_S_partial) - BTn_flat
    EM_cost_output['err_max_B'] = np.max(np.abs(B_err))
    EM_cost_output['max_j'] = np.max(np.linalg.norm(np.einsum(
        'oijk,kdij,ij,o->ijd', matrixd_phi, S.dpsi, 1/S.dS, j_S, optimize=True), axis=2))
    EM_cost_output['cost_B'] = Np * \
        np.einsum('i,i,i->', B_err, B_err, plasma_dS_normalized)
    EM_cost_output['cost_J'] = Np*np.einsum('i,ij,j->', j_S, Qj, j_S)
    return EM_cost_output


def EM_cost_dask(config, S, Sp):
    """new version without Lagrange multipliers, to use by default

    :param config: 
    :type config: :class:`configparser.ConfigParser`
    :param S:
    :type S: `Surface`
    :param Sp:
    :type Sp: `Surface`
    :return: various component of the cost
    :rtype: dictionary
    """
    import dask.array as da
    # initilization of the parameters
    lamb = float(config['other']['lamb'])
    Np = int(config['geometry']['Np'])
    ntheta_plasma = int(config['geometry']['ntheta_plasma'])
    ntheta_coil = int(config['geometry']['ntheta_coil'])
    nzeta_plasma = int(config['geometry']['nzeta_plasma'])
    nzeta_coil = int(config['geometry']['nzeta_coil'])
    mpol_coil = int(config['geometry']['mpol_coil'])
    ntor_coil = int(config['geometry']['ntor_coil'])
    net_poloidal_current_Amperes = float(
        config['other']['net_poloidal_current_Amperes'])/Np  # 11884578.094260072
    net_toroidal_current_Amperes = float(
        config['other']['net_toroidal_current_Amperes'])  # 0
    curpol = float(config['other']['curpol'])  # 4.9782004309255496
    phisize = (mpol_coil, ntor_coil)
    # 'code/li383/plasma_surf.txt'
    path_plasma = str(config['geometry']['path_plasma'])
    path_cws = str(config['geometry']['path_cws'])  # 'code/li383/cws.txt'
    path_bnorm = str(config['other']['path_bnorm'])  # 'code/li383/bnorm.txt'
    path_output = str(config['other']['path_output'])  # 'coeff_full_opt'
    cupy = config['other']['cupy'] == 'True'  # dask is needed to use cupy

    chunk_theta_coil = int(config['dask_parameters']['chunk_theta_coil'])
    chunk_zeta_coil = int(config['dask_parameters']['chunk_zeta_coil'])
    chunk_theta_plasma = int(config['dask_parameters']['chunk_theta_plasma'])
    chunk_zeta_plasma = int(config['dask_parameters']['chunk_zeta_plasma'])
    chunk_theta = int(config['dask_parameters']['chunk_theta'])

    # initialization of the surfaces
    if S is None:
        S_parametrization = Surface_Fourier.load_file(path_cws)
        S = Surface_Fourier(S_parametrization, (ntheta_coil, nzeta_coil), Np)
    if Sp is None:
        Sp_parametrization = Surface_Fourier.load_file(path_plasma)
        Sp = Surface_Fourier(Sp_parametrization,
                             (ntheta_plasma, nzeta_plasma), Np)

    if cupy:
        import cupy as cp
        def f(x): return x.map_blocks(cp.asarray, dtype=np.float64)
        def f_np(x): return cp.asarray(x)
        def get(x): return x.get()
    else:
        f, f_np, get = lambda x: x, lambda x: x, lambda x: x
    # tensors computations
    rot_tensor = tools.get_rot_tensor(Np)
    T = tools.get_tensor_distance(S, Sp, rot_tensor)
    rot_tensor = f_np(rot_tensor)
    T = f(da.from_array(T, chunks=(3, chunk_theta_coil, chunk_zeta_coil,
          chunk_theta_plasma, chunk_zeta_plasma, 3), asarray=False))
    matrixd_phi = f(da.from_array(tools.get_matrix_dPhi(phisize, S.grids), chunks={
                    1: chunk_theta_coil, 2: chunk_zeta_coil}, asarray=False))
    dpsi = f(da.from_array(S.dpsi, chunks=(
        2, 3, chunk_theta_coil, chunk_zeta_coil), asarray=False))
    normalp = f(da.from_array(Sp.n, chunks=(
        3, chunk_theta_plasma, chunk_zeta_plasma), asarray=False))
    S_dS = f(da.from_array(S.dS, chunks=(
        chunk_theta_coil, chunk_zeta_coil), asarray=False))
    D = 1/(da.linalg.norm(T, axis=-1)**3)
    DD = 1/(da.linalg.norm(T, axis=-1)**5)
    eijk = f_np(tools.eijk)
    Qj = tools.compute_Qj(matrixd_phi, dpsi, S_dS)
    K = np.einsum('sijpqa,sijpq->sijpqa', T, D)
    LS = (mu_0/(4*np.pi))*contract('sijpqa,tijh,sbc,hcij,dab,dpq->tpq', K, matrixd_phi,
                                   rot_tensor, dpsi, eijk, normalp, optimize=True)/(ntheta_coil*nzeta_coil)

    Qj = get(Qj.compute())
    LS = get(LS.compute())
    BT = -curpol*bnorm.get_bnorm(path_bnorm, Sp)
    # Regcoil:
    EM_cost_output = {}
    # WARNING : we restrict our space to hangle a constraint free pb
    LS_R = LS[2:]
    Qj_inv_R = np.linalg.inv(Qj[2:, 2:])
    LS_dagger_R = np.einsum('ut,tij,ij->uij', Qj_inv_R, LS_R, Sp.dS/Sp.npts)
    inside_M_lambda_R = lamb * \
        np.eye(LS_R.shape[0])+np.einsum('tpq,upq->tu', LS_dagger_R, LS_R)
    M_lambda_R = np.linalg.inv(inside_M_lambda_R)

    # we compute the full Right Hand Side
    B_tilde = BT - \
        np.einsum('tpq,t', LS[:2], [
                  net_poloidal_current_Amperes, net_toroidal_current_Amperes])
    LS_dagger_B_tilde = np.einsum('hpq,pq->h', LS_dagger_R, B_tilde)
    RHS = LS_dagger_B_tilde-lamb * \
        Qj_inv_R@Qj[2:, :2]@[net_poloidal_current_Amperes,
                             net_toroidal_current_Amperes]
    j_S_R = M_lambda_R@RHS
    j_S = np.concatenate(
        ([net_poloidal_current_Amperes, net_toroidal_current_Amperes], j_S_R))

    # we save the results
    B_err = np.einsum('hpq,h', LS, j_S) - BT
    EM_cost_output['err_max_B'] = np.max(np.abs(B_err))
    EM_cost_output['max_j'] = np.max(np.linalg.norm(np.einsum(
        'oijk,kdij,ij,o->ijd', get(matrixd_phi.compute()), S.dpsi, 1/S.dS, j_S, optimize=True), axis=2))
    EM_cost_output['cost_B'] = Np * \
        np.einsum('pq,pq,pq->', B_err, B_err, Sp.dS/Sp.npts)
    EM_cost_output['cost_J'] = Np*np.einsum('i,ij,j->', j_S, Qj, j_S)
    EM_cost_output['cost'] = EM_cost_output['cost_B'] + \
        lamb*EM_cost_output['cost_J']
    # Added for visualization
    j_3D = np.einsum('a,aijm,mlij->ijl', j_S, matrixd_phi, dpsi)
    EM_cost_output['j_3D'] = j_3D
    return EM_cost_output


def EM_cost_dask_with_multipliers(config, S, Sp):
    import dask.array as da
    # initilization of the parameters
    lamb = float(config['other']['lamb'])
    Np = int(config['geometry']['Np'])
    ntheta_plasma = int(config['geometry']['ntheta_plasma'])
    ntheta_coil = int(config['geometry']['ntheta_coil'])
    nzeta_plasma = int(config['geometry']['nzeta_plasma'])
    nzeta_coil = int(config['geometry']['nzeta_coil'])
    mpol_coil = int(config['geometry']['mpol_coil'])
    ntor_coil = int(config['geometry']['ntor_coil'])
    net_poloidal_current_Amperes = float(
        config['other']['net_poloidal_current_Amperes'])/Np  # 11884578.094260072
    net_toroidal_current_Amperes = float(
        config['other']['net_toroidal_current_Amperes'])  # 0
    curpol = float(config['other']['curpol'])  # 4.9782004309255496
    phisize = (mpol_coil, ntor_coil)
    # 'code/li383/plasma_surf.txt'
    path_plasma = str(config['geometry']['path_plasma'])
    path_cws = str(config['geometry']['path_cws'])  # 'code/li383/cws.txt'
    path_bnorm = str(config['other']['path_bnorm'])  # 'code/li383/bnorm.txt'
    path_output = str(config['other']['path_output'])  # 'coeff_full_opt'
    cupy = config['other']['cupy'] == 'True'  # dask is needed to use cupy

    chunk_theta_coil = int(config['dask_parameters']['chunk_theta_coil'])
    chunk_zeta_coil = int(config['dask_parameters']['chunk_zeta_coil'])
    chunk_theta_plasma = int(config['dask_parameters']['chunk_theta_plasma'])
    chunk_zeta_plasma = int(config['dask_parameters']['chunk_zeta_plasma'])
    chunk_theta = int(config['dask_parameters']['chunk_theta'])

    # initialization of the surfaces
    if S is None:
        S_parametrization = Surface_Fourier.load_file(path_cws)
        S = Surface_Fourier(S_parametrization, (ntheta_coil, nzeta_coil), Np)
    if Sp is None:
        Sp_parametrization = Surface_Fourier.load_file(path_plasma)
        Sp = Surface_Fourier(Sp_parametrization,
                             (ntheta_plasma, nzeta_plasma), Np)

    if cupy:
        import cupy as cp
        def f(x): return x.map_blocks(cp.asarray, dtype=np.float64)
        def f_np(x): return cp.asarray(x)
        def get(x): return x.get()
    else:
        f, f_np, get = lambda x: x, lambda x: x, lambda x: x
    # tensors computations
    rot_tensor = tools.get_rot_tensor(Np)
    T = tools.get_tensor_distance(S, Sp, rot_tensor)
    rot_tensor = f_np(rot_tensor)
    T = f(da.from_array(T, chunks=(3, chunk_theta_coil, chunk_zeta_coil,
          chunk_theta_plasma, chunk_zeta_plasma, 3), asarray=False))
    matrixd_phi = f(da.from_array(tools.get_matrix_dPhi(phisize, S.grids), chunks={
                    1: chunk_theta_coil, 2: chunk_zeta_coil}, asarray=False))
    dpsi = f(da.from_array(S.dpsi, chunks=(
        2, 3, chunk_theta_coil, chunk_zeta_coil), asarray=False))
    normalp = f(da.from_array(Sp.n, chunks=(
        3, chunk_theta_plasma, chunk_zeta_plasma), asarray=False))
    S_dS = f(da.from_array(S.dS, chunks=(
        chunk_theta_coil, chunk_zeta_coil), asarray=False))
    D = 1/(da.linalg.norm(T, axis=-1)**3)
    DD = 1/(da.linalg.norm(T, axis=-1)**5)
    eijk = f_np(tools.eijk)
    Qj = tools.compute_Qj(matrixd_phi, dpsi, S_dS)
    K = np.einsum('sijpqa,sijpq->sijpqa', T, D)
    LS = (mu_0/(4*np.pi))*contract('sijpqa,tijh,sbc,hcij,dab,dpq->tpq', K, matrixd_phi,
                                   rot_tensor, dpsi, eijk, normalp, optimize=True)/(ntheta_coil*nzeta_coil)

    Qj = get(Qj.compute())
    LS = get(LS.compute())
    BT = -curpol*bnorm.get_bnorm(path_bnorm, Sp)
    # Regcoil:
    EM_cost_output = {}
    # WARNING : LS contained the poloidal and toroidal currents coordinates
    Qj_inv = np.linalg.inv(Qj)
    LS_dagger = np.einsum('ut,tij,ij->uij', Qj_inv, LS, Sp.dS/Sp.npts)
    inside_M_lambda = lamb * \
        np.eye(len(Qj[0]))+np.einsum('tpq,upq->tu', LS_dagger, LS)
    M_lambda = np.linalg.inv(inside_M_lambda)

    # we compute the 2 Lagrange multipliers
    LS_dagger_B_T = np.einsum('hpq,pq->h', LS_dagger, BT)
    m0, m1 = np.linalg.inv(M_lambda[:2]@Qj_inv[:, :2]) @ ((M_lambda@LS_dagger_B_T)[
        :2] - [net_poloidal_current_Amperes, net_toroidal_current_Amperes])
    RHS = LS_dagger_B_T - [m0, m1]@Qj_inv[:2, :]
    j_S = M_lambda@RHS

    # we save the results

    B_err = np.einsum('hpq,h', LS, j_S) - BT
    EM_cost_output['err_max_B'] = np.max(np.abs(B_err))
    EM_cost_output['max_j'] = np.max(np.linalg.norm(np.einsum(
        'oijk,kdij,ij,o->ijd', get(matrixd_phi.compute()), S.dpsi, 1/S.dS, j_S, optimize=True), axis=2))
    EM_cost_output['cost_B'] = Np * \
        np.einsum('pq,pq,pq->', B_err, B_err, Sp.dS/Sp.npts)
    EM_cost_output['cost_J'] = Np*np.einsum('i,ij,j->', j_S, Qj, j_S)
    return EM_cost_output


def EM_cost_dask_old(config, S, Sp):
    """old version"""
    import dask.array as da
    # initilization of the parameters
    lamb = float(config['other']['lamb'])
    Np = int(config['geometry']['Np'])
    ntheta_plasma = int(config['geometry']['ntheta_plasma'])
    ntheta_coil = int(config['geometry']['ntheta_coil'])
    nzeta_plasma = int(config['geometry']['nzeta_plasma'])
    nzeta_coil = int(config['geometry']['nzeta_coil'])
    mpol_coil = int(config['geometry']['mpol_coil'])
    ntor_coil = int(config['geometry']['ntor_coil'])
    net_poloidal_current_Amperes = float(
        config['other']['net_poloidal_current_Amperes'])/Np  # 11884578.094260072
    net_toroidal_current_Amperes = float(
        config['other']['net_toroidal_current_Amperes'])  # 0
    curpol = float(config['other']['curpol'])  # 4.9782004309255496
    phisize = (mpol_coil, ntor_coil)
    # 'code/li383/plasma_surf.txt'
    path_plasma = str(config['geometry']['path_plasma'])
    path_cws = str(config['geometry']['path_cws'])  # 'code/li383/cws.txt'
    path_bnorm = str(config['other']['path_bnorm'])  # 'code/li383/bnorm.txt'
    path_output = str(config['other']['path_output'])  # 'coeff_full_opt'
    cupy = config['other']['cupy'] == 'True'  # dask is needed to use cupy

    chunk_theta_coil = int(config['dask_parameters']['chunk_theta_coil'])
    chunk_zeta_coil = int(config['dask_parameters']['chunk_zeta_coil'])
    chunk_theta_plasma = int(config['dask_parameters']['chunk_theta_plasma'])
    chunk_zeta_plasma = int(config['dask_parameters']['chunk_zeta_plasma'])
    chunk_theta = int(config['dask_parameters']['chunk_theta'])

    # initialization of the surfaces
    if S is None:
        S_parametrization = Surface_Fourier.load_file(path_cws)
        S = Surface_Fourier(S_parametrization, (ntheta_coil, nzeta_coil), Np)
    if Sp is None:
        Sp_parametrization = Surface_Fourier.load_file(path_plasma)
        Sp = Surface_Fourier(Sp_parametrization,
                             (ntheta_plasma, nzeta_plasma), Np)

    if cupy:
        import cupy as cp
        def f(x): return x.map_blocks(cp.asarray, dtype=np.float64)
        def f_np(x): return cp.asarray(x)
        def get(x): return x.get()
    else:
        f, f_np, get = lambda x: x, lambda x: x, lambda x: x
    # tensors computations
    rot_tensor = tools.get_rot_tensor(Np)
    T = tools.get_tensor_distance(S, Sp, rot_tensor)
    rot_tensor = f_np(rot_tensor)
    T = f(da.from_array(T, chunks=(3, chunk_theta_coil, chunk_zeta_coil,
          chunk_theta_plasma, chunk_zeta_plasma, 3), asarray=False))
    matrixd_phi = f(da.from_array(tools.get_matrix_dPhi(phisize, S.grids), chunks={
                    1: chunk_theta_coil, 2: chunk_zeta_coil}, asarray=False))
    dpsi = f(da.from_array(S.dpsi, chunks=(
        2, 3, chunk_theta_coil, chunk_zeta_coil), asarray=False))
    normalp = f(da.from_array(Sp.n, chunks=(
        3, chunk_theta_plasma, chunk_zeta_plasma), asarray=False))
    S_dS = f(da.from_array(S.dS, chunks=(
        chunk_theta_coil, chunk_zeta_coil), asarray=False))
    D = 1/(da.linalg.norm(T, axis=-1)**3)
    DD = 1/(da.linalg.norm(T, axis=-1)**5)
    eijk = f_np(tools.eijk)
    Qj = tools.compute_Qj(matrixd_phi, dpsi, S_dS)
    K = np.einsum('sijpqa,sijpq->sijpqa', T, D)
    LS = (mu_0/(4*np.pi))*contract('sijpqa,tijh,sbc,hcij,dab,dpq->tpq', K, matrixd_phi,
                                   rot_tensor, dpsi, eijk, normalp, optimize=True)/(ntheta_coil*nzeta_coil)

    Qj = get(Qj.compute())
    LS = get(LS.compute())
   # if cupy :
   #    LS=LS.get() #to obtain an np.array
   # else:
   #    LS=tools.compute_LS(T,matrixd_phi,S.dpsi,rot_tensor,Sp.n)
    array_bnorm = curpol*bnorm.get_bnorm(path_bnorm, Sp)
    # Regcoil:
    EM_cost_output = {}
    LS_matrix = np.transpose(np.reshape(
        LS[2:, :, :], (LS.shape[0]-2, -1)))  # matrix shape
    BTn = net_poloidal_current_Amperes * \
        LS[0]+net_toroidal_current_Amperes*LS[1]+array_bnorm
    # - for 2 reasons: we use inward convention (thus -array_bnorm)
    BTn_flat = -BTn.flatten()
    # and we want to eliminate the effect of the net currents

    # we take the adjoint, note that we have to take care of the different innerproduct
    plasma_dS_normalized = Sp.dS.flatten()/(Sp.nbpts[0]*Sp.nbpts[1])
    inside_M_lambda = np.einsum('ij,i,ik->jk', LS_matrix, plasma_dS_normalized, LS_matrix)\
        + lamb*Qj[2:, 2:]
    RHS = (np.einsum('ij,i,i->j', LS_matrix, plasma_dS_normalized, BTn_flat)-lamb *
           (net_poloidal_current_Amperes*Qj[2:, 0]+net_toroidal_current_Amperes*Qj[2:, 1]))
    j_S_partial = np.linalg.solve(inside_M_lambda, RHS)
    j_S = np.concatenate(
        ([net_poloidal_current_Amperes, net_toroidal_current_Amperes], j_S_partial))

    # we save the results

    B_err = (LS_matrix @ j_S_partial) - BTn_flat
    EM_cost_output['err_max_B'] = np.max(np.abs(B_err))
    EM_cost_output['max_j'] = np.max(np.linalg.norm(np.einsum(
        'oijk,kdij,ij,o->ijd', get(matrixd_phi.compute()), S.dpsi, 1/S.dS, j_S, optimize=True), axis=2))
    EM_cost_output['cost_B'] = Np * \
        np.einsum('i,i,i->', B_err, B_err, plasma_dS_normalized)
    EM_cost_output['cost_J'] = Np*np.einsum('i,ij,j->', j_S, Qj, j_S)
    EM_cost_output['cost'] = EM_cost_output['cost_B'] + \
        lamb*EM_cost_output['cost_J']
    return EM_cost_output
