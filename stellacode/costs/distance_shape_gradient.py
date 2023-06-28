import configparser
import logging

import stellacode.tools as tools
from stellacode import np
from stellacode.costs.abstract_shape_gradient import Abstract_shape_gradient
from stellacode.costs.auxi import f_non_linear
from stellacode.surface.surface_Fourier import Surface_Fourier
from stellacode.surface.surface_from_file import surface_from_file


class Distance_shape_gradient(Abstract_shape_gradient):
    """Non linear penalization of the distance to the plasma (lower bound)"""

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        # Preparation of the plasma and geometric properties
        self.Np = int(config["geometry"]["Np"])
        ntheta_plasma = int(config["geometry"]["ntheta_plasma"])
        nzeta_plasma = int(config["geometry"]["nzeta_plasma"])
        # 'code/li383/plasma_surf.txt'
        path_plasma = str(config["geometry"]["path_plasma"])
        self.Sp = surface_from_file(path_plasma, n_fp=self.Np, n_pol=ntheta_plasma, n_tor=nzeta_plasma)
        self.rot_tensor = tools.get_rot_tensor(self.Np)
        # distance cost parameters
        self.d_min_hard = float(config["optimization_parameters"]["d_min_hard"])
        self.d_min_soft = float(config["optimization_parameters"]["d_min_soft"])
        self.d_min_penalization = float(config["optimization_parameters"]["d_min_penalization"])

        self.vf = np.vectorize(lambda x: f_non_linear(self.d_min_hard, self.d_min_soft, self.d_min_penalization, x))

    def cost(self, S):
        T = tools.get_tensor_distance(S, self.Sp, self.rot_tensor)
        dist = np.linalg.norm(T, axis=-1)
        dist_min = np.amin(dist, axis=(0, 3, 4))
        cost = self.Np * np.einsum("ij,ij->", self.vf(dist_min), S.dS / S.npts)

        return cost, {"min_distance": np.min(dist_min)}
