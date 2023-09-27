import configparser

import jax
import jax.numpy as np
import numpy as onp
from jax import grad

from stellacode.costs.aggregate_cost import AggregateCost
from stellacode.surface.imports import get_cws


def test_full_grad_against_original_np_grad():
    jax.config.update("jax_enable_x64", True)

    config = configparser.ConfigParser()
    config.read("config_file/config_small.ini")

    full_grad = AggregateCost.from_config(config)
    factory = get_cws(config)

    def fun(par):
        factory.update_params(**par)
        metrics = full_grad.cost(factory())[1]

        return metrics["cost_B"] + full_grad.costs[0].lamb * 3 * metrics["cost_J"]

    cost = fun(factory.get_trainable_params())
    assert onp.abs(cost - 0.04940551129939265) < 1e-15

    gradf = grad(fun)
    grad_res = gradf(factory.get_trainable_params())
    grad_res_np = np.load("data/full_shape_grad.npy")
    onp.testing.assert_array_almost_equal(grad_res["0.Rmn"], grad_res_np[: grad_res["0.Rmn"].shape[0]])
    onp.testing.assert_array_almost_equal(grad_res["0.Zmn"], grad_res_np[grad_res["0.Zmn"].shape[0] :])
