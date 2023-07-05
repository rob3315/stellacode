import configparser

import jax
import jax.numpy as np
import numpy as onp
from jax import grad

from stellacode.costs.aggregate_cost import AggregateCost


def test_full_grad_against_original_np_grad():
    jax.config.update("jax_enable_x64", True)

    config = configparser.ConfigParser()
    config.read("config_file/config_small.ini")

    full_grad = AggregateCost(config=config)
    fun = lambda par: full_grad.cost(**par)

    cost = full_grad.cost(**full_grad.init_param)
    assert onp.abs(cost - 0.04940551129939265) < 1e-15

    gradf = grad(fun)
    grad_res = gradf(full_grad.init_param)
    grad_res_np = np.load("data/full_shape_grad.npy")
    onp.testing.assert_array_almost_equal(
        grad_res["Rmn"], grad_res_np[: grad_res["Rmn"].shape[0]]
    )
    onp.testing.assert_array_almost_equal(
        grad_res["Zmn"], grad_res_np[grad_res["Zmn"].shape[0] :]
    )
