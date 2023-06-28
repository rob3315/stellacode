def chiB_v_chiK(lambdas, path_config=None, config=None):
    """For several values of lambda, computes chi²_B and chi²_K.

    :param lambdas: array of lambda values
    :type lambdas: 1D array

    :param path_config: path to configuration file (.ini). Default : None.
    :type path_config: str

    :param config: configuration. Default : None.
    :type config: configparser.ConfigParser

    :return: list of chi_squared_B, list of chi_squared_K
    :rtype: tuple(list, list)
    """
    from configparser import ConfigParser

    import dask.array as da
    from opt_einsum import contract
    from scipy.constants import mu_0

    import stellacode.tools as tools
    import stellacode.tools.bnorm as bnorm
    from stellacode import np
    from stellacode.surface.surface_from_file import surface_from_file

    if config is None:
        config = ConfigParser()
        config.read(path_config)

    # Initializing the parameters
    lamb = float(config["other"]["lamb"])
    Np = int(config["geometry"]["Np"])
    ntheta_plasma = int(config["geometry"]["ntheta_plasma"])
    ntheta_coil = int(config["geometry"]["ntheta_coil"])
    nzeta_plasma = int(config["geometry"]["nzeta_plasma"])
    nzeta_coil = int(config["geometry"]["nzeta_coil"])
    mpol_coil = int(config["geometry"]["mpol_coil"])
    ntor_coil = int(config["geometry"]["ntor_coil"])
    net_poloidal_current_Amperes = float(config["other"]["net_poloidal_current_Amperes"]) / Np
    net_toroidal_current_Amperes = float(config["other"]["net_toroidal_current_Amperes"])
    curpol = float(config["other"]["curpol"])
    phisize = (mpol_coil, ntor_coil)
    path_plasma = str(config["geometry"]["path_plasma"])
    path_cws = str(config["geometry"]["path_cws"])
    path_bnorm = str(config["other"]["path_bnorm"])

    chunk_theta_coil = int(config["dask_parameters"]["chunk_theta_coil"])
    chunk_zeta_coil = int(config["dask_parameters"]["chunk_zeta_coil"])
    chunk_theta_plasma = int(config["dask_parameters"]["chunk_theta_plasma"])
    chunk_zeta_plasma = int(config["dask_parameters"]["chunk_zeta_plasma"])
    chunk_theta = int(config["dask_parameters"]["chunk_theta"])

    # initialization of the surfaces
    S = surface_from_file(path_cws, Np, ntheta_coil, nzeta_coil)
    Sp = surface_from_file(path_plasma, Np, ntheta_plasma, nzeta_plasma)

    # tensors computations
    BT = curpol * bnorm.get_bnorm(path_bnorm, Sp)

    rot_tensor = tools.get_rot_tensor(Np)
    matrixd_phi = tools.get_matrix_dPhi(phisize, S.grids)
    dpsi = S.dpsi
    normalp = Sp.n
    S_dS = S.dS
    eijk = tools.eijk
    Qj = tools.compute_Qj(matrixd_phi, dpsi, S_dS)

    def compute_B(XYZgrid):
        T = (
            XYZgrid[np.newaxis, np.newaxis, np.newaxis, ...]
            - contract("opq,ijq->oijp", rot_tensor, S.P)[..., np.newaxis, np.newaxis, :]
        )

        K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

        B = (
            mu_0
            / (4 * np.pi)
            * contract("nuvabj,ouvk,niq,kquv,ijc->oabc", K, matrixd_phi, rot_tensor, dpsi, eijk)
            / (ntheta_coil * nzeta_coil)
        )
        return B

    XYZgrid_dask = da.from_array(Sp.P, chunks=(chunk_theta_plasma, chunk_zeta_plasma, 3))

    B = da.map_blocks(
        compute_B,
        XYZgrid_dask,
        dtype=np.float64,
        chunks=(len(matrixd_phi) // 10, chunk_theta_plasma, chunk_zeta_plasma, 3),
    ).compute()

    LS = contract("oabc,cab->oab", B, normalp)

    Qj = Qj
    LS = LS
    # Inward
    BT = curpol * bnorm.get_bnorm(path_bnorm, Sp)

    # WARNING : we restrict our space to hangle a constraint free pb
    LS_R = LS[2:]
    Qj_inv_R = np.linalg.inv(Qj[2:, 2:])
    LS_dagger_R = np.einsum("ut,tij,ij->uij", Qj_inv_R, LS_R, Sp.dS / Sp.npts)

    # Initializing the results
    chi_squared_Bs = []
    chi_squared_Ks = []

    for lamb in lambdas:
        inside_M_lambda_R = lamb * np.eye(LS_R.shape[0]) + np.einsum("tpq,upq->tu", LS_dagger_R, LS_R)
        M_lambda_R = np.linalg.inv(inside_M_lambda_R)

        # we compute the full Right Hand Side
        B_tilde = BT - np.einsum(
            "tpq,t",
            LS[:2],
            [net_poloidal_current_Amperes, net_toroidal_current_Amperes],
        )
        LS_dagger_B_tilde = np.einsum("hpq,pq->h", LS_dagger_R, B_tilde)
        RHS = LS_dagger_B_tilde - lamb * Qj_inv_R @ Qj[2:, :2] @ [
            net_poloidal_current_Amperes,
            net_toroidal_current_Amperes,
        ]
        j_S_R = M_lambda_R @ RHS
        j_S = np.concatenate(([net_poloidal_current_Amperes, net_toroidal_current_Amperes], j_S_R))

        B_err = np.einsum("hpq,h", LS, j_S) - BT

        # cost_B
        chi_squared_Bs.append(Np * np.einsum("pq,pq,pq->", B_err, B_err, Sp.dS / Sp.npts))

        # cost_J
        chi_squared_Ks.append(Np * np.einsum("i,ij,j->", j_S, Qj, j_S))

    return chi_squared_Bs, chi_squared_Ks
