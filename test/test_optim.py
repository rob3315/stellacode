import jax

from stellacode.optimizer import Optimizer

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from stellacode.costs import (
    AggregateCost,
    CurrentCtrCost,
    DistanceCost,
    EMCost,
    NegTorCurvatureCost,
)
from stellacode.costs.utils import Constraint
from stellacode.definitions import w7x_plasma
from stellacode.optimizer import Optimizer
from stellacode.surface import IntegrationParams
from stellacode.surface.factories import get_pwc_surface, get_toroidal_surface


def test_optimization():
    path_config_file = "test/data/li383/config.ini"

    opt = Optimizer.from_config_file(path_config_file)
    opt.max_iter = 1
    opt.method = "L-BFGS-B"
    opt.options = {"disp": True, "maxls": 1}
    opt.optimize()


@pytest.mark.parametrize("surface_name", ["axisym", "pwc"])
def test_current_optim(surface_name):
    n_harmonics = 4
    factor = 6
    num_points = n_harmonics * factor
    method = "quadratic"

    em_cost = EMCost.from_plasma_config(
        plasma_config=w7x_plasma,
        integration_par=IntegrationParams(num_points_u=num_points, num_points_v=num_points),
        use_mu_0_factor=False,
        train_currents=True,
    )

    distance = DistanceCost(
        Sp=em_cost.Sp,
        constraint=Constraint(limit=0.2, distance=0.03, minimum=True, method=method),
    )
    current_ctr = CurrentCtrCost(constraint=Constraint(limit=9, distance=0.3, minimum=False, method=method))
    neg_curv = NegTorCurvatureCost(constraint=Constraint(limit=-0.05, distance=0.1, minimum=True, method=method))
    agg_cost = AggregateCost(costs=[em_cost, distance, neg_curv, current_ctr])
    if surface_name == "axisym":
        coil_surf = get_toroidal_surface(
            em_cost.Sp,
            plasma_path=w7x_plasma.path_plasma,
            n_harmonics=n_harmonics,
            factor=factor,
            distance=0.5,
        )
    else:
        coil_surf = get_pwc_surface(
            em_cost.Sp,
            plasma_path=w7x_plasma.path_plasma,
            n_harmonics=n_harmonics,
            factor=factor,
            distance=0.5,
            rotate_diff_current=3,
            common_current_on_each_rot=True,
        )

    opt = Optimizer.from_cost(
        agg_cost,
        coil_surf,
        method="L-BFGS-B",
        kwargs=dict(options={"disp": False, "maxls": 10, "maxiter": 5}),
    )

    cost, metrics, results, optimized_params = opt.optimize()

    metrics["deltaB_B"] = np.sqrt(metrics["cost_B"] / 1056)
    assert metrics["deltaB_B"] < 0.15
