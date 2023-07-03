import configparser

import jax

jax.config.update("jax_enable_x64", True)
from stellacode import np
from scipy.io import netcdf_file
from stellacode.costs.EM_cost import EMCost
from stellacode.surface.imports import get_cws


def test_no_dimension_error():
    ###just check that all operations respects dimensions
    path_config_file = "config_file/config_test_dim.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    EMCost.from_config_file(path_config_file)


def test_compare_to_regcoil():
    path_config_file = "test/data/li383/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)

    filename = "test/data/li383/regcoil_out.li383.nc"
    file_ = netcdf_file(filename, "r", mmap=False)

    cws = get_cws(config)
    em_cost = EMCost.from_config(config=config)
    lambdas = np.array([1.2e-14, 1.0e00])
    metrics = em_cost.cost_multiple_lambdas(cws, lambdas)

    chi2_b = file_.variables["chi2_B"][()][1:].astype(float)
    assert np.max(np.abs(metrics.cost_B.values - chi2_b) / chi2_b) < 5e-5
    chi_j = file_.variables["chi2_K"][()][1:].astype(float)
    assert np.max(np.abs(metrics.cost_J.values - chi_j) / chi_j) < 5e-6
