import configparser

import jax
import pandas as pd

from stellacode.costs.EM_cost import EMCost
from stellacode.surface.imports import get_cws
from stellacode.surface.cylindrical import CylindricalSurface

jax.config.update("jax_enable_x64", True)
def to_float(dict_ ):return {k: float(v) for k, v in dict_.items()}

def regcoil_vs_lambda(config, lambdas):
    em_cost = EMCost.from_config(config=config)
    em_cost.net_currents = None
    em_cost.use_mu_0_factor = False
    # S = get_cws(config)
    # S = CylindricalSurface()
    fourier_coeffs = np.zeros((5,2))
    # fourier_coeffs[-1, 0] = 0.5
    S = CylindricalSurface(params={"fourier_coeffs": fourier_coeffs}, nbpts=(64,64), Np=3)

    BS = em_cost.get_BS_norm(S)
    results = {}
    for lamb in lambdas:
        j_S, Qj = em_cost.get_current(BS=BS, S=S, lamb=lamb)
        results[lamb] = to_float(em_cost.get_results(BS=BS, j_S=j_S, S=S, Qj=Qj, lamb=lamb)[1])

        # j_3D = em_cost.get_j_3D(j_S=j_S, S=S)

    return pd.DataFrame(results)


# path_config = "/home_nfs/bruno.rigal/wsp/stellacode-loris/config_loris/config_hsr4_conformal.ini"
# path_config = "/home_nfs/bruno.rigal/wsp/stellacode-loris/config_loris/config_w7x_conformal.ini"
path_config = "/home_nfs/bruno.rigal/wsp/stellacode/config_file/config_full.ini" # li383
lambdas = [10 ** i for i in range(-1, -31, -1)]
config = configparser.ConfigParser()
config.read(path_config)


em_cost = EMCost.from_config(config=config)

import pdb;pdb.set_trace()
fourier_coeffs = np.zeros((5,2))
S = CylindricalSurface(params={"fourier_coeffs": fourier_coeffs}, nbpts=(64,64), Np=3)

# res = regcoil_vs_lambda(config, lambdas)
# print(res.T.min())
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(15, 10))
# res.T.plot("max_j", "rmse_B_norm", ax=ax)
# ax.set_xlabel(r"max J [A]")
# ax.set_ylabel(r"$rmse(B_norm)$")
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.set_title("conformal HSR4")
# plt.savefig("lambd.png")


# import pdb;pdb.set_trace()
