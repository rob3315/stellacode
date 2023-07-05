import configparser

import jax
import pandas as pd
import utilitiesRF as urf

from stellacode import np
from stellacode.costs.em_cost import EMCost
from stellacode.surface.cylindrical import CylindricalSurface
from stellacode.surface.imports import get_cws

jax.config.update("jax_enable_x64", True)

from stellacode.tools.vmec import VMECIO

vmec = VMECIO("test/data/li383/wout_li383_1.4m.nc")
val = vmec.get_val("b", 0.1, 0.1, 1)

import pdb

pdb.set_trace()


def to_float(dict_):
    return {k: float(v) for k, v in dict_.items()}


def regcoil_vs_lambda(config, lambdas):
    em_cost = EMCost.from_config(config=config)
    # em_cost.net_currents = None
    em_cost.use_mu_0_factor = False
    # S = get_cws(config)
    # S = CylindricalSurface()
    fourier_coeffs = np.zeros((5, 2))
    # fourier_coeffs[-1, 0] = 0.5
    S = CylindricalSurface(params={"fourier_coeffs": fourier_coeffs}, nbpts=(64, 64), Np=3)

    BS = em_cost.get_BS_norm(S)
    results = {}
    for lamb in lambdas:
        j_S, Qj = em_cost.get_current(BS=BS, S=S, lamb=lamb)
        results[lamb] = to_float(em_cost.get_results(BS=BS, j_S=j_S, S=S, Qj=Qj, lamb=lamb)[1])

        vmec = urf.VmecIO()
        vmec.read_wout("test/data/li383/wout_li383_1.4m.nc")
        vmec.fields(nradius=49, ntheta=64, nzeta=64)

        b_average = np.mean(em_cost.Sp.num_tor_symmetry * vmec.B[1] ** 2 * em_cost.Sp.dS)

        results[lamb]["B/dB"] = results[lamb]["cost_B"] / b_average

        # from stellacode.tools.vmec import VMECIO

        # vmec = VMECIO('test/data/li383/wout_li383_1.4m.nc')

        # # vmec.get_val("b", 0.1, 0.1, 1)

        #
        # vmec = urf.VmecIO()
        # vmec.read_wout('test/data/li383/wout_li383_1.4m.nc')
        # vmec.fields(nradius=49, ntheta=64,nzeta=64)

        # B = np.stack((vmec.Bx[:,:,:], vmec.By[:,:,:], vmec.BZ[:,:,:]), axis=-1)
        # np.mean(np.linalg.norm(B, axis=-1))

        # vmec.modulusB()
        # em_cost.bnorm-vmec.B_s

        # np.mean(np.abs(em_cost.bnorm+vmec.B_s[-1]))/np.mean(np.abs(vmec.B_s[-1]))
        # vmec.B_s
        # import pdb;pdb.set_trace()
        # j_3D = em_cost.get_j_3D(j_S=j_S, S=S)

    return pd.DataFrame(results)


# path_config = "/home_nfs/bruno.rigal/wsp/stellacode-loris/config_loris/config_hsr4_conformal.ini"
# path_config = "/home_nfs/bruno.rigal/wsp/stellacode-loris/config_loris/config_w7x_conformal.ini"
path_config = "/home/bruno/wsp/stellacode/config_file/config_full.ini"  # li383
lambdas = [10**i for i in range(-1, -31, -1)]
config = configparser.ConfigParser()
config.read(path_config)


# em_cost = EMCost.from_config(config=config)

res = regcoil_vs_lambda(config, lambdas)
import pdb

pdb.set_trace()
fourier_coeffs = np.zeros((5, 2))
S = CylindricalSurface(params={"fourier_coeffs": fourier_coeffs}, nbpts=(64, 64), Np=3)

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
