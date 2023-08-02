from stellacode.surface import ToroidalSurface

# def test_axisymmetric():
#     surf = ToroidalSurface(nbpts=(16,16))

#     filename = "test/data/li383/regcoil_out.li383.nc"
#     file_ = netcdf_file(filename, "r", mmap=False)

#     cws = get_cws(config)
#     xm, xn = cws.current.get_coeffs()
#     assert np.all(file_.variables["xm_coil"][()][1:] == xm)
#     assert np.all(file_.variables["xn_coil"][()][1:] // 3 == xn)

#     em_cost = EMCost.from_config(config=config, use_mu_0_factor=use_mu_0_factor)
