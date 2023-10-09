import configparser

import jax
import numpy as np
import pandas as pd

from stellacode.costs.em_cost import EMCost

# from stellacode.surface.utils import get_cws
from stellacode.surface.cylindrical import CylindricalSurface

# path_config = "/home_nfs/bruno.rigal/wsp/stellacode-loris/config_loris/config_hsr4_conformal.ini"
# path_config = "/home_nfs/bruno.rigal/wsp/stellacode-loris/config_loris/config_w7x_conformal.ini"
path_config = "/home/bruno/wsp/stellacode/config_file/config_full.ini"  # li383
lambdas = [10**i for i in range(-1, -31, -1)]
config = configparser.ConfigParser()
config.read(path_config)


em_cost = EMCost.from_config(config=config)

# tor_pos = em_cost.Sp.cartesian_to_toroidal()

from stellacode.surface.fourier import FourierSurfaceFactory
from stellacode.surface.tore import ToroidalSurface
from stellacode.tools.utils import get_min_dist

res = em_cost.Sp.cartesian_to_toroidal()
minor_radius = np.max(res[:, :, 0])
major_radius = em_cost.Sp.params["Rmn"][0]
# surf = ToroidalSurface(Np=3, major_radius = major_radius, minor_radius=minor_radius+0.2, params={}, nbpts=(64,64))
# get_min_dist(surf.P, em_cost.Sp.xyz)

# # surf = FourierSurface(Np=1, mf=np.array([0]), nf=np.array([0]), params={"Zmn":np.array([0]), "Rmn":np.array([1]) }, nbpts=(32,32))
# res = surf.cartesian_to_toroidal(surf.xyz)
# em_cost.Sp.cartesian_to_toroidal()


fourier_coeffs = np.zeros((5, 2))
S = CylindricalSurface(
    params={"fourier_coeffs": fourier_coeffs},
    radius=minor_radius + 0.1294 + 1.0,
    distance=major_radius,
    nbpts=(64, 64),
    nfp=6,
)
get_min_dist(S.get_rotated_xyz(), em_cost.Sp.xyz)

import pdb

pdb.set_trace()
from mayavi import mlab

S.plot(only_one_period=True)
em_cost.Sp.plot(only_one_period=True)
mlab.show()

S.plot()
em_cost.Sp.plot()
mlab.show()

import pdb

pdb.set_trace()


xyz = em_cost.Sp.P


np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)


fourier_coeffs = np.zeros((5, 2))
S = CylindricalSurface(params={"fourier_coeffs": fourier_coeffs}, nbpts=(64, 64), Np=3)

uv_grid = np.stack(em_cost.Sp.grids, axis=0)
cyl_pos = em_cost.Sp.cartesian_to_toroidal(em_cost.Sp.xyz)

em_cost.Sp.params["Rmn"]

# S.plot(only_one_period=True)
# em_cost.Sp.plot(only_one_period=True)

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
