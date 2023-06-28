import configparser
import logging

from stellacode import np
from stellacode.costs.abstract_shape_gradient import Abstract_shape_gradient
from stellacode.costs.auxi import f_e
from stellacode.surface.surface_Fourier import Surface_Fourier


class Curvature_shape_gradient(Abstract_shape_gradient):
    """Non linear penalization on the curvature (upper bound)"""

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        self.Np = int(config["geometry"]["Np"])
        self.ntheta_coil = int(config["geometry"]["ntheta_coil"])
        self.nzeta_coil = int(config["geometry"]["nzeta_coil"])
        self.c0 = float(config["optimization_parameters"]["curvature_c0"])
        self.c1 = float(config["optimization_parameters"]["curvature_c1"])
        # self.f = lambda x: f_e(self.c0, self.c1, np.maximum(x, 0.))
        self.gf = lambda x: grad_f_e(self.c0, self.c1, np.maximum(x, 0.0))
        self.f = np.vectorize(lambda x: f_e(self.c0, self.c1, np.maximum(x, 0.0)))
        # self.gf = np.vectorize(lambda x: grad_f_e(self.c0, self.c1, np.maximum(x, 0.)))

    def cost(self, S):
        pmax, pmin = S.principles[0], S.principles[1]
        # f_pmax = np.array([[self.f(elt) for elt in x] for x in pmax])
        # f_pmin = np.array([[self.f(-elt) for elt in x] for x in pmin])
        f_pmax = self.f(pmax)  # np.array([[self.f(elt) for elt in x] for x in pmax])
        f_pmin = self.f(pmin)  # np.array([[self.f(-elt) for elt in x] for x in pmin])
        cost = self.Np * np.einsum("ij,ij->", f_pmax, S.dS / S.npts)
        cost += self.Np * np.einsum("ij,ij->", f_pmin, S.dS / S.npts)
        aux_dic = {}
        aux_dic["max_curvature"] = max(np.max(pmax), np.max(-pmin))
        # logging.info(
        #     'maximal curvature {:5e} m^-1, curvature cost : {:5e}'.format(aux_dic['max_curvature'], cost))
        return cost, aux_dic
