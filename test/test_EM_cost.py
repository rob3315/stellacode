import configparser

import jax
import numpy as onp

jax.config.update("jax_enable_x64", True)

from stellacode.costs.EM_cost import EMCost
from stellacode.surface.utils import get_cws


def test_no_dimension_error():
    ###just check that all operations respects dimensions
    path_config_file = "config_file/config_test_dim.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    EMCost.from_config_file(path_config_file)


def test_compare_to_regcoil():
    path_config_file = "config_file/config.ini"
    config = configparser.ConfigParser()
    config.read(path_config_file)
    # data from laplace_code
    err_max_B = [1.26945824e-02, 2.67485601e-01, 5.42332579e-02, 1.73186188e-02]
    max_j = [9.15419611e07, 3.94359068e06, 7.35442879e06, 1.52081420e07]
    cost_B = [3.49300835e-04, 1.36627491e-01, 5.56293801e-03, 4.67141080e-04]
    cost_J = [8.96827408e15, 1.00021438e14, 1.42451562e14, 6.65195111e14]

    cws = get_cws(config)
    config["other"]["dask"] = "False"
    for index, lamb in enumerate([0, 1.2e-14, 2.5e-16, 5.1e-19]):
        config["other"]["lamb"] = str(lamb)
        em_cost = EMCost.from_config(config=config)
        EM_cost, EM_cost_output = em_cost.cost(cws)

        onp.testing.assert_almost_equal(EM_cost_output["err_max_B"], err_max_B[index])
        onp.testing.assert_almost_equal(EM_cost_output["max_j"], max_j[index], decimal=-1)
        onp.testing.assert_almost_equal(EM_cost_output["cost_B"], cost_B[index])
        onp.testing.assert_almost_equal(EM_cost_output["cost_J"], cost_J[index], decimal=-9)
