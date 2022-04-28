import logging
import numpy as np
import configparser
from src.surface.surface_Fourier import Surface_Fourier
from src.costs.abstract_shape_gradient import Abstract_shape_gradient
from src.costs.auxi import f_e, grad_f_e


class Curvature_shape_gradient(Abstract_shape_gradient):
    """Non linear penalization on the curvature (upper bound)
    """

    def __init__(self, path_config_file=None, config=None):
        if config is None:
            config = configparser.ConfigParser()
            config.read(path_config_file)
        self.config = config
        self.Np = int(config['geometry']['Np'])
        self.ntheta_coil = int(config['geometry']['ntheta_coil'])
        self.nzeta_coil = int(config['geometry']['nzeta_coil'])
        self.c0 = float(config['optimization_parameters']['curvature_c0'])
        self.c1 = float(config['optimization_parameters']['curvature_c1'])
        self.f = lambda x: f_e(self.c0, self.c1, np.max((x, 0.)))
        self.gf = lambda x: grad_f_e(self.c0, self.c1, np.max((x, 0.)))

    def cost(self, S):
        pmax, pmin = S.principles[0], S.principles[1]
        f_pmax = np.array([[self.f(elt) for elt in x] for x in pmax])
        f_pmin = np.array([[self.f(-elt) for elt in x] for x in pmin])
        cost = self.Np*np.einsum('ij,ij->', f_pmax, S.dS/S.npts)
        cost += self.Np*np.einsum('ij,ij->', f_pmin, S.dS/S.npts)
        aux_dic = {}
        aux_dic['max_curvature'] = max(np.max(pmax), np.max(-pmin))
        logging.info(
            'maximal curvature {:5e} m^-1, curvature cost : {:5e}'.format(aux_dic['max_curvature'], cost))
        return cost, aux_dic

    def curvature_derivative(self, S, theta_peturbation):
        dtheta = theta_peturbation['dtheta']
        result = {}
        dE = 2*np.einsum('lij,oijl->oij', S.dpsi[0], dtheta[:, :, :, 0, :])
        dF = np.einsum('lij,oijl->oij', S.dpsi[0], dtheta[:, :, :, 1, :])+np.einsum(
            'lij,oijl->oij', S.dpsi[1], dtheta[:, :, :, 0, :])
        dG = 2*np.einsum('lij,oijl->oij', S.dpsi[1], dtheta[:, :, :, 1, :])
        (E, F, G) = S.I
        result['dI'] = (dE, dF, dG)
        (d2theta_uu, d2theta_uv, d2theta_vv) = theta_peturbation['d2theta']
        dndtheta = theta_peturbation['dndtheta']
        dL = np.einsum('oijl,lij->oij', d2theta_uu, S.n) + \
            np.einsum('lij,oijl->oij', S.dpsi_uu, dndtheta)  # e
        dM = np.einsum('oijl,lij->oij', d2theta_uv, S.n) + \
            np.einsum('lij,oijl->oij', S.dpsi_uv, dndtheta)  # f
        dN = np.einsum('oijl,lij->oij', d2theta_vv, S.n) + \
            np.einsum('lij,oijl->oij', S.dpsi_vv, dndtheta)  # g
        result['dII'] = (dL, dM, dN)
        L, M, N = S.II

        det1 = (E*G-F**2)
        det2 = (L*N-M**2)
        ddet1 = dE*G+dG*E-2*dF*F
        ddet2 = dL*N+dN*L-2*dM*M
        K = (L*N-M**2)/(E*G-F**2)
        dK = ddet2/det1-ddet1*det2/(det1)**2

        # trace of (second fundamental)(first fundamental^-1)
        # Mean Curvature
        H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
        up = (E*N + G*L - 2*F*M)
        dup = dE*N+dN*E+dG*L+dL*G-2*dF*M-2*dM*F
        dH = (dup/det1 - ddet1*up/(det1)**2)/2
        result['dK'] = dK
        result['dH'] = dH
        dPmax = dH + (dH*H-0.5*dK)/np.sqrt(H**2 - K)
        dPmin = dH - (dH*H-0.5*dK)/np.sqrt(H**2 - K)
        result['dPmax'] = dPmax
        result['dPmin'] = dPmin
        return result

    def shape_gradient(self, S, theta_pertubation):
        result_curvature_derivative = self.curvature_derivative(
            S, theta_pertubation)
        dPmax = result_curvature_derivative['dPmax']
        dPmin = result_curvature_derivative['dPmin']
        pmax, pmin = S.principles[0], S.principles[1]
        dSdtheta = theta_pertubation['dSdtheta']
        grad_f_pmax = np.array([[self.gf(elt) for elt in x] for x in pmax])
        f_pmax = np.array([[self.f(elt) for elt in x] for x in pmax])
        grad_f_pmin = np.array([[self.gf(-elt) for elt in x] for x in pmin])
        f_pmin = np.array([[self.f(-elt) for elt in x] for x in pmin])
        grad = self.Np*np.einsum('ij,oij,ij->o', grad_f_pmax, dPmax, S.dS/S.npts) + \
            self.Np*np.einsum('ij,oij->o', f_pmax, dSdtheta/S.npts)
        grad += self.Np*np.einsum('ij,oij,ij->o', grad_f_pmin, -dPmin, S.dS/S.npts) + \
            self.Np*np.einsum('ij,oij->o', f_pmin, dSdtheta/S.npts)
        return grad
