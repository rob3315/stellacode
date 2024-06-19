from os.path import join

import configparser

import jax
import jax.numpy as np
import numpy as onp
from jax import grad

from stellacode.costs.aggregate_cost import AggregateCost
from stellacode.surface.imports import get_cws
from stellacode import PROJECT_PATH


def test_full_grad_against_original_np_grad():
    """
    Test the gradients of the cost function against the original numpy gradients.
    """

    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    # Read the configuration file
    config = configparser.ConfigParser()
    config_file_folder = join(PROJECT_PATH, "config_file")
    config.read(join(config_file_folder, "config_small.ini"))

    # Create the aggregate cost object
    full_grad = AggregateCost.from_config(config)

    # Create the factory object
    factory = get_cws(config)

    # Define the function to be differentiated
    def fun(par):
        """
        Function to be differentiated.

        Args:
            par (dict): The parameters of the factory object.

        Returns:
            float: The cost value.
        """
        factory.update_params(**par)
        metrics = full_grad.cost(factory())[1]

        return metrics["cost_B"] + full_grad.costs[0].lamb * 3 * metrics["cost_J"]

    # Compute the cost value
    cost = fun(factory.get_trainable_params())

    # Assert that the cost value is close to the original numpy value
    assert onp.abs(cost - 0.04940551129939265) < 1e-15

    # Compute the gradients
    gradf = grad(fun)
    grad_res = gradf(factory.get_trainable_params())

    # Load the original numpy gradients
    configs_folder = join(PROJECT_PATH, "data")
    grad_res_np = np.load(join(configs_folder, "full_shape_grad.npy"))

    # Assert that the gradients are almost equal to the original numpy gradients
    onp.testing.assert_array_almost_equal(
        grad_res["0.Rmn"], grad_res_np[: grad_res["0.Rmn"].shape[0]])
    onp.testing.assert_array_almost_equal(
        grad_res["0.Zmn"], grad_res_np[grad_res["0.Zmn"].shape[0]:])
