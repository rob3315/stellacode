import configparser
import logging
import os
import pickle
from typing import Any, Optional

import jax
from pydantic import BaseModel, Extra

from stellacode.costs.abstract_cost import Results
from stellacode.costs.aggregate_cost import AggregateCost
from stellacode.surface.coil_surface import CoilSurface
from stellacode.surface.imports import get_cws
from stellacode.tools.concat_dict import ScaleDictArray
from autograd_minimize import minimize

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
    """transform a dict in a string"""
    out_str = ""
    for k, val in res_dict.items():
        # if isinstance(val, int):
        out_str += f"{k}: {{{k}}}, "
        # else:
        #     out_str += f"{k}: {{{k}:.4f}}, "
    return out_str


class Optimizer(BaseModel):
    cost: AggregateCost
    coil_surface: CoilSurface
    loss: Any
    init_param: Any
    scaler: Optional[ScaleDictArray] = None
    freq_save: int = 100
    save_res: bool = False
    output_folder_name: Optional[str] = None
    info: dict = dict(Nfeval=0)
    method: Optional[str] = None
    kwargs: dict = {}

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow  # allow extra fields

    @classmethod
    def from_config_file(cls, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        return cls.from_config(config)

    @classmethod
    def from_config(cls, config):
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

        info = {"Nfeval": 0}

        # optimizer options
        freq_save = int(config["optimization_parameters"]["freq_save"])
        max_iter = int(config["optimization_parameters"]["max_iter"])

        return cls.from_cost(
            cost=cost,
            coil_surface=cws,
            info=info,
            save_res=save_res,
            freq_save=freq_save,
            options={"maxiter": max_iter},
        )

    @classmethod
    def from_cost(
        cls,
        cost,
        coil_surface,
        set_scales: bool = False,
        preset_scales: dict = {},
        freeze_params: list = [],
        **kwargs,
    ):
        init_param = coil_surface.get_trainable_params()
        init_param = {k: v for k, v in init_param.items() if k not in freeze_params}
        if set_scales:
            scaler = ScaleDictArray(scales=preset_scales)
            init_param = scaler.apply(init_param)
        else:
            scaler = None

        def loss(**kwargs):
            if scaler is not None:
                kwargs = scaler.unapply(kwargs)
            # tic = time()
            coil_surface.update_params(**kwargs)
            # print("Surface", time() - tic)

            res, metrics, results = cost.cost(coil_surface, results=Results())

            jax.debug.print(tostr_jax(metrics), **metrics)
            # print(metrics)
            # log_info(
            #     info,
            #     res,
            #     freq_save=freq_save,
            #     save_res=save_res,
            #     output_folder_name=output_folder_name,
            # )
            return res

        return cls(
            cost=cost,
            coil_surface=coil_surface,
            scaler=scaler,
            loss=loss,
            init_param=init_param,
            **kwargs,
        )

    def optimize(self):
        # The optimization
        optim_res = minimize(self.loss, self.init_param, method=self.method, backend="jax", **self.kwargs)

        if self.save_res:
            logging.warning("optimization ended, saving file")
            with open("{}/result".format(self.output_folder_name), "wb") as output_file:
                pickle.dump(optim_res, output_file)

        # assert optimize_shape.success
        optimized_params = optim_res.x

        self.coil_surface.update_params(**optimized_params)
        cost, metrics, results = self.cost.cost(self.coil_surface)
        return cost, metrics, results, optimized_params


def log_info(info, res, freq_save=100, save_res=False, output_folder_name=""):
    # display information
    neval = info["Nfeval"]
    if neval % freq_save == 0:
        logging.info(f"Neval : {0:4d} \n saving the intermediate shape".format(neval))
        if save_res:
            with open(
                f"{output_folder_name}/intermediate{neval}.res",
                "wb",
            ) as output_file:
                pickle.dump(res, output_file)
    info["Nfeval"] += 1
