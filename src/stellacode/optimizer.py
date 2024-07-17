import configparser
import logging
import os
import pickle
from typing import Any, Callable, Optional

import jax
from autograd_minimize import minimize
from pydantic import BaseModel, Extra

from stellacode.costs.abstract_cost import Results
from stellacode.costs.aggregate_cost import AggregateCost
from stellacode.surface.abstract_surface import AbstractBaseFactory
from stellacode.surface.imports import get_cws
from stellacode.tools.concat_dict import ScaleDictArray

logger = logging.getLogger(__name__)


def tostr(res_dict):
    """transform a dict in a string"""
    out_str = ""
    for k, val in res_dict.items():
        if isinstance(val, int):
            out_str += f"{k}: {val}, "
        else:
            out_str += f"{k}: {val:.4f}, "
    return out_str


def tostr_jax(res_dict):
    """transform a dict in a string adapted for printing with jax."""
    out_str = ""
    for k, val in res_dict.items():
        out_str += f"{k}: {{{k}}}, "
    return out_str


class Optimizer(BaseModel):
    """
    Optimize a given surface with a list of costs.

    Args:
        * cost: cost to be optimized
        * coil_factory: coil factory with the parameters to be optimized
        * loss: loss function to be optimized
        * init_param: initial parameters for the optimization
        * scaler: scale the parameter beofre running the optimization
        * save_res: save the result
        * output_folder_name: folder in which the result is saved
        * method: method used for the optimization (see scipy minimize)
        * kwargs: dict of arguments passed to the optimizer
    """

    cost: AggregateCost
    coil_factory: AbstractBaseFactory
    loss: Callable
    init_param: dict
    scaler: Optional[ScaleDictArray] = None
    save_res: bool = False
    output_folder_name: Optional[str] = None
    method: Optional[str] = None
    kwargs: dict = {}

    model_config = dict(arbitrary_types_allowed=True)

    @classmethod
    def from_config_file(cls, config_file):
        """
        Create an instance of `Optimizer` from a configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            `Optimizer`: An instance of `Optimizer`.
        """
        # Read the configuration file
        config = configparser.ConfigParser()
        config.read(config_file)

        # Create an instance of `Optimizer` from the configuration
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of `Optimizer` from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            `Optimizer`: An instance of `Optimizer`.
        """
        # We create the temporary folder
        save_res = config["other"]["path_output"] != "None"
        if save_res:
            output_folder_name = "tmp/" + config["other"]["path_output"]
            os.makedirs(output_folder_name, exist_ok=True)
            # We save the config file there
            with open("{}/config.ini".format(output_folder_name), "w") as tmp_file:
                config.write(tmp_file)
        else:
            output_folder_name = None

        cost = AggregateCost.from_config(config)
        cws = get_cws(config)

        # optimizer options
        max_iter = int(config["optimization_parameters"]["max_iter"])

        return cls.from_cost(
            cost=cost,
            coil_factory=cws,
            save_res=save_res,
            options={"maxiter": max_iter},
        )

    @classmethod
    def from_cost(
        cls,
        cost,
        coil_factory,
        set_scales: bool = False,
        preset_scales: dict = {},
        freeze_params: list = [],
        **kwargs,
    ):
        """
        Create an instance of `Optimizer` from a cost and a coil factory.

        Args:
            cost (AggregateCost): Cost to be optimized.
            coil_factory (AbstractBaseFactory): Coil factory with the parameters to be optimized.
            set_scales (bool, optional): Set the scales of the parameters for the optimization. Defaults to False.
            preset_scales (dict, optional): Scales to be applied to the parameters for the optimization. Defaults to {}.
            freeze_params (list, optional): Parameters that should not be optimized. Defaults to [].
            **kwargs: Additional arguments for the optimization.

        Returns:
            Optimizer: An instance of `Optimizer`.
        """
        init_param = coil_factory.get_trainable_params()
        init_param = {k: v for k, v in init_param.items()
                      if k not in freeze_params}

        if set_scales:
            scaler = ScaleDictArray(scales=preset_scales)
            init_param = scaler.apply(init_param)
        else:
            scaler = None

        def loss(**kwargs):
            """
            Compute the loss function.

            Args:
                **kwargs: Parameters for the optimization.

            Returns:
                float: The loss.
            """
            if scaler is not None:
                kwargs = scaler.unapply(kwargs)

            coil_factory.update_params(**kwargs)
            res, metrics, results, S = cost.cost(
                coil_factory(), results=Results())
            jax.debug.print(tostr_jax(metrics), **metrics)

            return res

        return cls(
            cost=cost,
            coil_factory=coil_factory,
            scaler=scaler,
            loss=loss,
            init_param=init_param,
            **kwargs,
        )

    def optimize(self):
        """
        Run the optimization.

        Returns:
            tuple: A tuple containing the cost, metrics, results and optimized params.
        """

        # The optimization
        optim_res = minimize(self.loss, self.init_param,
                             method=self.method, backend="jax", **self.kwargs)

        # Save the result if specified
        if self.save_res:
            logging.warning("optimization ended, saving file")
            with open(f"{self.output_folder_name}/result", "wb") as output_file:
                pickle.dump(optim_res, output_file)

        # Get the optimized parameters
        optimized_params = optim_res.x

        # Update the coil factory with the optimized parameters
        self.coil_factory.update_params(**optimized_params)

        # Compute the cost, metrics and results
        cost, metrics, results, S = self.cost.cost(self.coil_factory())

        # Return the cost, metrics, results and optimized params
        return cost, metrics, results, optimized_params


def log_info(info, res, freq_save=100, save_res=False, output_folder_name=""):
    # display information
    neval = info["Nfeval"]
    if neval % freq_save == 0:
        logging.info(
            f"Neval : {0:4d} \n saving the intermediate shape".format(neval))
        if save_res:
            with open(
                f"{output_folder_name}/intermediate{neval}.res",
                "wb",
            ) as output_file:
                pickle.dump(res, output_file)
    info["Nfeval"] += 1
