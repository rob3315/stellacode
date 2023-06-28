import configparser
import logging

from opt_einsum import contract
from scipy.constants import mu_0

import stellacode.tools as tools
import stellacode.tools.bnorm as bnorm
from stellacode import np
from stellacode.costs.abstract_shape_gradient import Abstract_shape_gradient
from stellacode.costs.EM_cost import EM_cost_dask_3
from stellacode.surface.surface_from_file import Surface_Fourier, surface_from_file


class EM_shape_gradient(Abstract_shape_gradient):
    """Main cost coming from the inverse problem"""

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        self.Np = int(config["geometry"]["Np"])
        ntheta_plasma = int(config["geometry"]["ntheta_plasma"])
        self.ntheta_coil = int(config["geometry"]["ntheta_coil"])
        nzeta_plasma = int(config["geometry"]["nzeta_plasma"])
        self.nzeta_coil = int(config["geometry"]["nzeta_coil"])

        self.lamb = float(config["other"]["lamb"])

        # 'code/li383/plasma_surf.txt'
        path_plasma = str(config["geometry"]["path_plasma"])
        path_cws = str(config["geometry"]["path_cws"])  # 'code/li383/cws.txt'

        # initialization of the surfaces
        self.Sp = surface_from_file(path_plasma, n_fp=self.Np, n_pol=ntheta_plasma, n_tor=nzeta_plasma)
        self.S = surface_from_file(path_cws, n_fp=self.Np, n_pol=self.ntheta_coil, n_tor=self.nzeta_coil)

        self.EM_cost = EM_cost_dask_3.from_config(config, self.Sp)

    def cost(self, S):
        EM_cost_dic = self.EM_cost.get_cost(S, self.Sp)
        EM_cost = EM_cost_dic["cost_B"] + self.lamb * EM_cost_dic["cost_J"]
        return EM_cost, EM_cost_dic
