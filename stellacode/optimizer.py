import configparser
import logging
import os
import pickle
from typing import Any, Optional

import jax
import scipy.optimize
from pydantic import BaseModel, Extra

from stellacode.costs.aggregate_cost import AggregateCost
from stellacode.tools.concat_dict import ConcatDictArray

logger = logging.getLogger(__name__)


class Optimizer(BaseModel):
    info: dict
    cost: AggregateCost
    save_res: bool
    concater: ConcatDictArray
    freq_save: int
    max_iter: int
    loss_and_grad: Any
    init_param: Any
    output_folder_name: Optional[str] = None

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

        cost = AggregateCost(config=config)

        info = {"Nfeval": 0}
        concater = ConcatDictArray()
        init_param = concater.concat(cost.init_param)

        # loss_and_grad = jax.value_and_grad(loss)

        # optimizer options
        freq_save = int(config["optimization_parameters"]["freq_save"])
        max_iter = int(config["optimization_parameters"]["max_iter"])

        def loss(X):
            kwargs = concater.unconcat(X)
            res = cost.cost(**kwargs)

            log_info(info, res, freq_save=freq_save, save_res=save_res, output_folder_name=output_folder_name)

            return res

        return cls(
            info=info,
            cost=cost,
            save_res=save_res,
            output_folder_name=output_folder_name,
            concater=concater,
            freq_save=freq_save,
            max_iter=max_iter,
            loss_and_grad=jax.value_and_grad(loss),
            init_param=init_param,
        )

    @profile
    def optimize(self):
        self.loss_and_grad(self.init_param)
        self.loss_and_grad(self.init_param)
        self.loss_and_grad(self.init_param)
        self.loss_and_grad(self.init_param)

        # The optimization
        optimize_shape = scipy.optimize.minimize(
            self.loss_and_grad,
            self.init_param,
            jac=True,
            options={"maxiter": self.max_iter, "return_all": True},
        )

        if self.save_res:
            logging.warning("optimization ended, saving file")
            with open("{}/result".format(self.output_folder_name), "wb") as output_file:
                pickle.dump(optimize_shape, output_file)


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
