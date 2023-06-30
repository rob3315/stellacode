import configparser
import logging
import os
import pickle
import sys

import jax
import scipy.optimize

from stellacode.costs.aggregate_cost import AggregateCost
from stellacode.optimizer import Optimizer
from stellacode.tools.concat_dict import ConcatDictArray

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6


# def launch(path_config_file=None, config=None):
#     if config is None:
#         config = configparser.ConfigParser()
#         config.read(path_config_file)

#     # We create the temporary folder
#     save_res = config["other"]["path_output"] != "None"
#     if save_res:
#         output_folder_name = "tmp/" + config["other"]["path_output"]
#         os.makedirs(output_folder_name, exist_ok=True)
#         # We save the config file there
#         with open("{}/config.ini".format(output_folder_name), "w") as tmp_file:
#             config.write(tmp_file)

#     full_grad = AggregateCost(config=config)

#     info = {"Nfeval": 0}
#     concater = ConcatDictArray()
#     concater.concat(full_grad.init_param)

#     def loss(X):
#         kwargs = concater.unconcat(X)
#         res = full_grad.cost(**kwargs)
#         # display information
#         if info["Nfeval"] % freq_save == 0:
#             logging.info("Neval : {0:4d} \n saving the intermediate shape".format(info["Nfeval"]))
#             if save_res:
#                 with open(
#                     "{}/intermediate{:}.res".format(output_folder_name, info["Nfeval"]),
#                     "wb",
#                 ) as output_file:
#                     pickle.dump(res, output_file)
#         info["Nfeval"] += 1
#         return res

#     loss_and_grad = jax.value_and_grad(loss)

#     # optimizer options
#     freq_save = int(config["optimization_parameters"]["freq_save"])
#     max_iter = int(config["optimization_parameters"]["max_iter"])


#     # The optimization
#     optimize_shape = scipy.optimize.minimize(
#         loss_and_grad,
#         full_grad.init_param,
#         jac=True,
#         options={"maxiter": max_iter, "return_all": True},
#     )

#     if save_res:
#         logging.warning("optimization ended, saving file")
#         with open("{}/result".format(output_folder_name), "wb") as output_file:
#             pickle.dump(optimize_shape, output_file)


if __name__ == "__main__":
    opt = Optimizer.from_config_file(sys.argv[1])
    opt.optimize()
    # launch(sys.argv[1])  # launch the script with the argument of the call python launcher.py path_to_config
    # launch('config_file/config_full.ini')
