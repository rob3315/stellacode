import numpy as np
from ..abstract_classes.abstract_pwc import PWC_Surface
import logging

PI = np.pi
float_type = np.float64


class Surface_PWC_Fourier(PWC_Surface):
    """A class used to represent an piecewise cylindrical surface.
    With the section being represented by Fourier coefficients.

    Parameters
    ----------

    n_fp : int
        Number of field periods.
    n_cyl : int
        Number of cylinders per field period. Can be equal to 1, or even, or a multiple of 4 if symmetry is True.
    symmetry : bool
        True if the Stellarator has Stellarator symmetry.
    R0 : float
        Radius of the Stellarator.
    l_u : int
        Number of poloidal angles per field period. Has to be even, and to be a multiple of n_cyl.
    l_v : int
        Number of toroidal angles.
    param : array
            parameters of the surface. Fourier coefficients and angles.
    """

    def __new__(cls, n_fp, n_cyl, symmetry, l_u, l_v, param):
        if l_v % 2 == 1:
            raise ValueError("l_v has to be even")
        elif n_cyl != 1 and n_cyl % 2 == 1:
            raise ValueError("n_cyl has to be equal to 1 or be even")
        elif l_v % n_cyl != 0:
            raise ValueError("l_v has to be a multiple of n_cyl")
        elif symmetry and n_cyl % 4 != 0 and n_cyl != 1:
            raise ValueError(
                "If symmetry, n_cyl has to be a multiple of 4 or be equal to 1")
        else:
            return super(Surface_PWC_Fourier, cls).__new__(cls)

    def __init__(self, n_fp, n_cyl, symmetry, l_u, l_v, param):
        if n_cyl == 1:
            param[-2] = PI * (0.5 - 1 / n_fp)
            param[-1] = 0

        self.__n_fp = n_fp
        self.__n_cyl = n_cyl
        self.__symmetry = symmetry
        self.l_u = l_u
        self.l_v = l_v
        self.__param = param

        self.__R0 = param[-3]
        self.__alpha = self.__param[-2]
        self.__beta = self.__param[-1]
        self.fourier_coeffs = self.__param[:-3:]

        # Computing the attributes of the surface
        self.__compute_r()
        self.compute_points()

        # First order attributes
        self.__compute_r_prime()
        self.compute_first_derivatives()

        # Second order attributes
        self.__compute_r_second()
        self.compute_second_derivatives()

    @classmethod
    def load_file(cls, path_surf, n_fp, n_pol, n_tor):
        """
        Creates a Surface_PWC object from a text file.
        """
        if n_pol % 4 == 0:
            print("WARNING : n_pol is a multiple of four. Some results may be wrong.")

        data = []
        with open(path_surf, 'r') as f:
            next(f)
            for line in f:
                data.append(str.split(line))

        n_cyl = int(data[0][0])
        symmetry = data[0][1] == "True"

        param = np.asarray(data[1], dtype=float_type)

        logging.debug('Fourier coefficients extracted from file')

        return cls(n_fp, n_cyl, symmetry, n_pol, n_tor, param)

    def _get_n_fp(self):
        """
        Number of field periods.
        """
        return self.__n_fp

    n_fp = property(_get_n_fp)

    def _get_n_cyl(self):
        """
        Number of field periods.
        """
        return self.__n_cyl

    n_cyl = property(_get_n_cyl)

    def _get_symmetry(self):
        return self.__symmetry

    symmetry = property(_get_symmetry)

    def _get_R0(self):
        return self.__R0

    R0 = property(_get_R0)

    def _get_r(self):
        return self.__r

    r = property(_get_r)

    def _get_r_prime(self):
        return self.__r_prime

    r_prime = property(_get_r_prime)

    def _get_r_second(self):
        return self.__r_second

    r_second = property(_get_r_second)

    def _get_alpha(self):
        return self.__alpha

    alpha = property(_get_alpha)

    def _get_beta(self):
        return self.__beta

    beta = property(_get_beta)

    def _get_npts(self):
        return self.l_u * self.l_v

    npts = property(_get_npts)

    def _get_nbpts(self):
        return (self.l_u, self.l_v)

    nbpts = property(_get_nbpts)

    def _get_grids(self):
        u, v = np.linspace(0, 1, self.l_u, endpoint=False), (np.arange(
            self.l_v) + 0.5) / self.l_v
        ugrid, vgrid = np.meshgrid(u, v, indexing='ij')
        return ugrid, vgrid

    grids = property(_get_grids)

    def _get_param(self):
        return self.__param

    def _set_param(self, param):
        self.__param = param
        self.__R0 = self.__param[-3]
        self.__alpha = self.__param[-2]
        self.__beta = self.__param[-1]
        self.fourier_coeffs = self.__param[:-3:]
        self.__compute_r()
        self.__compute_r_prime()
        self.__compute_r_second()
        self.compute_points()
        self.compute_first_derivatives()
        self.compute_second_derivatives()

    param = property(_get_param, _set_param)

    def __compute_r(self):
        """
        Returns a function which describes the cross section of the cylinder.

        Parameters
        ----------

        fourier_coeffs : array
            2D array. The n-th row has the two Fourier coefficients a_n and b_n.
        """
        def r(theta):
            n = len(self.fourier_coeffs) // 2
            return np.sum(self.fourier_coeffs * np.concatenate((np.cos(theta * np.arange(n)), np.sin(theta * np.arange(n)))))
        self.__r = np.vectorize(r)

    def __compute_r_prime(self):
        def r_prime(theta):
            n = len(self.fourier_coeffs) // 2
            coeffs = np.concatenate(
                (self.fourier_coeffs[1:len(self.fourier_coeffs)//2:], self.fourier_coeffs[1+len(self.fourier_coeffs)//2::]))
            return np.sum(coeffs * np.concatenate((- np.arange(1, n) * np.sin(theta * np.arange(1, n)), np.arange(1, n) * np.cos(theta * np.arange(1, n)))))
        self.__r_prime = np.vectorize(r_prime)

    def __compute_r_second(self):
        def r_second(theta):
            n = len(self.fourier_coeffs) // 2
            coeffs = np.concatenate(
                (self.fourier_coeffs[1:len(self.fourier_coeffs)//2:], self.fourier_coeffs[1+len(self.fourier_coeffs)//2::]))
            return np.sum(coeffs * np.concatenate((- np.arange(1, n)**2 * np.cos(theta * np.arange(1, n)), - np.arange(1, n)**2 * np.sin(theta * np.arange(1, n)))))
        self.__r_second = np.vectorize(r_second)

    def get_theta_pertubation(self, compute_curvature=True):
        """
        Compute the perturbations of a surface
        """
        us = np.linspace(0, 1, self.l_u, endpoint=False, dtype=float_type)
        vs = (np.arange(self.l_v // self.n_cyl, dtype=float_type) + 0.5) / self.l_v
        ugrid, vgrid = np.meshgrid(us, vs, indexing='ij')

        thetagrid = 2 * PI * ugrid
        phigrid = 2 * PI / self.n_fp * vgrid

        n_coeffs = len(self.fourier_coeffs)
        n = n_coeffs // 2
        perturbation = np.empty(
            (n_coeffs + 3, *ugrid.shape, 3), dtype=float_type)

        for k in range(n):
            perturbation[k, :, :, 0] = np.cos(
                phigrid) / np.sin(phigrid + self.alpha) * np.cos(thetagrid) * np.cos(k * thetagrid)
            perturbation[k, :, :, 1] = np.tan(
                phigrid) * perturbation[k, :, :, 0]
            perturbation[k, :, :, 2] = np.tan(
                self.beta) * perturbation[k, :, :, 1] + np.sin(thetagrid) * np.cos(k * thetagrid) / np.cos(self.beta)
            perturbation[k + n, :, :,
                         0] = np.tan(k * thetagrid) * perturbation[k, :, :, 0]
            perturbation[k + n, :, :,
                         1] = np.tan(k * thetagrid) * perturbation[k, :, :, 1]
            perturbation[k + n, :, :,
                         2] = np.tan(k * thetagrid) * perturbation[k, :, :, 2]

        perturbation[-3, :, :, 0] = np.tan(self.alpha) / \
            (np.tan(self.alpha) + np.tan(phigrid))
        perturbation[-3, :, :, 1] = np.tan(phigrid) * perturbation[-3, :, :, 0]
        perturbation[-3, :, :,
                     2] = np.tan(self.beta) * perturbation[-3, :, :, 1]

        perturbation[-2, :, :, 0] = self.R0 * (1 + np.tan(self.alpha)**2) * np.tan(
            phigrid) / (np.tan(self.alpha) + np.tan(phigrid))**2 - np.cos(phigrid) * np.cos(phigrid + self.alpha) / np.sin(phigrid + self.alpha)**2 * np.cos(thetagrid) * self.r(thetagrid)
        perturbation[-2, :, :, 1] = np.tan(phigrid) * perturbation[-2, :, :, 0]
        perturbation[-2, :, :,
                     2] = np.tan(self.beta) * perturbation[-2, :, :, 1]

        perturbation[-1, :, :, 0] = 0
        perturbation[-1, :, :, 1] = 0
        perturbation[-1, :, :, 2] = (1 + np.tan(self.beta)**2) * self.P[::, :self.l_v // self.n_cyl:, 1] + np.sin(
            self.beta) / np.cos(self.beta)**2 * np.sin(thetagrid) * self.r(thetagrid)

        dperturbation = np.empty(
            (n_coeffs + 3, *ugrid.shape, 2, 3))

        for k in range(n):
            # derivatives / phi
            dperturbation[k, :, :, 1, 0] = - np.cos(self.alpha) / np.sin(
                phigrid + self.alpha)**2 * np.cos(thetagrid) * np.cos(k * thetagrid)
            dperturbation[k, :, :, 1, 1] = (
                1 + np.tan(phigrid)**2) * perturbation[k, :, :, 0] + np.tan(phigrid) * dperturbation[k, :, :, 1, 0]
            dperturbation[k, :, :, 1, 2] = np.tan(
                self.beta) * dperturbation[k, :, :, 1, 1]

            dperturbation[k + n, :, :, 1,
                          0] = np.tan(k * thetagrid) * dperturbation[k, :, :, 1, 0]
            dperturbation[k + n, :, :, 1,
                          1] = np.tan(k * thetagrid) * dperturbation[k, :, :, 1, 1]
            dperturbation[k + n, :, :, 1,
                          2] = np.tan(k * thetagrid) * dperturbation[k, :, :, 1, 2]

            # derivatives / theta
            dperturbation[k, :, :, 0, 0] = - np.cos(phigrid) / np.sin(phigrid + self.alpha) * (
                np.sin(thetagrid) * np.cos(k * thetagrid) + k * np.cos(thetagrid) * np.sin(k * thetagrid))
            dperturbation[k, :, :, 0, 1] = np.tan(
                phigrid) * dperturbation[k, :, :, 0, 0]
            dperturbation[k, :, :, 0, 2] = np.tan(self.beta) * dperturbation[k, :, :, 0, 1] + 1 / np.cos(
                self.beta) * (np.cos(thetagrid) * np.cos(k * thetagrid) - k * np.sin(thetagrid) * np.sin(k * thetagrid))

            dperturbation[k + n, :, :, 0, 0] = np.tan(k * thetagrid) * dperturbation[k, :, :, 0, 0] + k * (
                1 + np.tan(k * thetagrid)**2) * perturbation[k, :, :, 0]
            dperturbation[k + n, :, :, 0, 1] = np.tan(k * thetagrid) * dperturbation[k, :, :, 0, 1] + k * (
                1 + np.tan(k * thetagrid)**2) * perturbation[k, :, :, 1]
            dperturbation[k + n, :, :, 0, 2] = np.tan(k * thetagrid) * dperturbation[k, :, :, 0, 2] + k * (
                1 + np.tan(k * thetagrid)**2) * perturbation[k, :, :, 2]

        # d²x / dphi dR0
        dperturbation[-3, :, :, 1, 0] = - np.tan(self.alpha) * (
            1 + np.tan(phigrid)**2) / (np.tan(self.alpha) + np.tan(phigrid))**2
        # d²y / dphi dR0
        dperturbation[-3, :, :, 1, 1] = (1 + np.tan(phigrid)**2) * perturbation[-3,
                                                                                :, :, 0] + np.tan(phigrid) * dperturbation[-3, :, :, 1, 0]
        # d²z / dphi dR0
        dperturbation[-3, :, :, 1,
                      2] = np.tan(self.beta) * dperturbation[-3, :, :, 1, 1]

        # d²x / dtheta dR0
        dperturbation[-3, :, :, 0, 0] = 0
        # d²y / dtheta dR0
        dperturbation[-3, :, :, 0, 1] = 0
        # d²z / dtheta dR0
        dperturbation[-3, :, :, 0, 2] = 0

        # d²x / dphi dalpha
        dperturbation[-2, :, :, 1, 0] = self.R0 * (1 + np.tan(self.alpha)**2) * (1 + np.tan(phigrid)**2) * (np.tan(self.alpha) - np.tan(phigrid)) / (np.tan(self.alpha) + np.tan(phigrid))**3 + (
            np.sin(2*phigrid + self.alpha) * np.sin(phigrid + self.alpha) + 2 * np.cos(phigrid) * np.cos(phigrid + self.alpha)**2) / np.sin(phigrid + self.alpha)**3 * np.cos(thetagrid) * self.r(thetagrid)
        # d²y / dphi dalpha
        dperturbation[-2, :, :, 1, 1] = (1 + np.tan(phigrid)**2) * perturbation[-2,
                                                                                :, :, 0] + np.tan(phigrid) * dperturbation[-2, :, :, 1, 0]
        # d²z / dphi dalpha
        dperturbation[-2, :, :, 1,
                      2] = np.tan(self.beta) * dperturbation[-2, :, :, 1, 1]

        # d²x / dtheta dalpha
        dperturbation[-2, :, :, 0, 0] = np.cos(phigrid) * np.cos(phigrid + self.alpha) / np.sin(
            phigrid + self.alpha)**2 * (np.sin(thetagrid) * self.r(thetagrid) - np.cos(thetagrid) * self.r_prime(thetagrid))
        # d²y / dtheta dalpha
        dperturbation[-2, :, :, 0,
                      1] = np.tan(phigrid) * dperturbation[-2, :, :, 0, 0]
        # d²z / dtheta dalpha
        dperturbation[-2, :, :, 0,
                      2] = np.tan(self.beta) * dperturbation[-2, :, :, 0, 1]

        # d²x / dphi dbeta
        dperturbation[-1, :, :, 1, 0] = 0
        # d²y / dphi dbeta
        dperturbation[-1, :, :, 1, 1] = 0
        # d²z / dphi dbeta
        dperturbation[-1, :, :, 1,
                      2] = (1 + np.tan(self.beta)**2) * self.dpsi[1, 1, ::, :self.l_v // self.n_cyl:] * self.n_fp / (2 * PI)

        # d²x / dtheta dbeta
        dperturbation[-1, :, :, 0, 0] = 0
        # d²y / dtheta dbeta
        dperturbation[-1, :, :, 0, 1] = 0
        # d²z / dtheta dbeta
        dperturbation[-1, :, :, 0, 2] = (1 + np.tan(self.beta)**2) * \
            self.dpsi[0, 1, ::, :self.l_v // self.n_cyl:] / (2 * PI) + np.sin(self.beta) / np.cos(self.beta)**2 * (
                np.cos(thetagrid) * self.r(thetagrid) + np.sin(thetagrid) * self.r_prime(thetagrid))

        # Conversion to u and v
        dperturbation[:, :, :, 0, :] *= 2 * PI
        dperturbation[:, :, :, 1, :] *= 2 * PI / self.n_fp

        other_half = np.copy(perturbation[::, ::, ::-1, ::])

        angle = 2 * np.pi / (self.n_fp * self.n_cyl)

        symmetry_matrix = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        np.einsum("ij,kuvj->kuvi", symmetry_matrix, other_half, out=other_half)

        two_cylinders_perturbation = np.concatenate(
            (perturbation, other_half), axis=2)
        full_perturbation = np.concatenate((perturbation, other_half), axis=2)

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ], dtype=float_type)

                np.einsum("ij,kuvj->kuvi", symmetry_matrix,
                          two_cylinders_perturbation[..., ::-1, :], out=two_cylinders_perturbation)
                full_perturbation = np.concatenate(
                    (full_perturbation, two_cylinders_perturbation), axis=2)
                angle += 4 * PI / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * PI / (self.n_fp * self.n_cyl)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=float_type)
            for _ in range(self.n_cyl // 2 - 1):
                np.einsum("ij,kuvj->kuvi", rotation_matrix,
                          two_cylinders_perturbation, out=two_cylinders_perturbation)
                full_perturbation = np.concatenate(
                    (full_perturbation, two_cylinders_perturbation), axis=2)

        dperturbation_sym = np.copy(dperturbation[::, ::, ::-1])

        angle = 2 * PI / (self.n_fp * self.n_cyl)

        sym_mat_1T = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        sym_mat_2T = np.array([
            [1, 0],
            [0, -1]
        ], dtype=float_type)

        np.einsum("ik,luvkj->luvij", sym_mat_2T,
                  dperturbation_sym, out=dperturbation_sym)
        np.einsum("luvik,kj->luvij", dperturbation_sym,
                  sym_mat_1T, out=dperturbation_sym)

        two_cylinders_dperturbation = np.concatenate(
            (dperturbation, dperturbation_sym), axis=2)

        full_dperturbation = np.concatenate(
            (dperturbation, dperturbation_sym), axis=2)

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            sym_mat_2T = np.array([
                [1, 0],
                [0, -1]
            ])

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrixT = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ])

                np.einsum("ik,luvkj->luvij", sym_mat_2T,
                          two_cylinders_dperturbation[::, ::, ::-1], out=two_cylinders_dperturbation)
                np.einsum("luvik,kj->luvij", two_cylinders_dperturbation,
                          symmetry_matrixT, out=two_cylinders_dperturbation)

                full_dperturbation = np.concatenate(
                    (full_dperturbation, two_cylinders_dperturbation), axis=2)

                angle += 4 * PI / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * PI / (self.n_fp * self.n_cyl)
            rotation_matrixT = np.array([
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for _ in range(self.n_cyl // 2 - 1):
                np.einsum("luvik,kj->luvij", two_cylinders_dperturbation,
                          rotation_matrixT, out=two_cylinders_dperturbation)

                full_dperturbation = np.concatenate(
                    (full_dperturbation, two_cylinders_dperturbation), axis=2)

        res = {}
        res['theta'] = full_perturbation
        res['dtheta'] = full_dperturbation

        dtilde_psi = np.array([self.dpsi[0], self.dpsi[1], self.n])
        partial_x_partial_u = np.linalg.inv(
            np.einsum('ijkl->klij', dtilde_psi))
        partial_x_partial_u_cut = partial_x_partial_u[:, :, :, :2]
        dtildetheta = np.einsum('ijkl,oijlm->oijkm',
                                partial_x_partial_u_cut, res['dtheta'])
        # for cross product
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        dNdtheta = np.einsum('dij,aije,def->aijf',
                             self.dpsi[0], res['dtheta'][:, :, :, 1, :], eijk)
        dNdtheta -= np.einsum('dij,aije,def->aijf',
                              self.dpsi[1], res['dtheta'][:, :, :, 0, :], eijk)
        dSdtheta = np.einsum('aijd,dij->aij', dNdtheta, self.N)/self.dS
        dndtheta = np.einsum('oijl,ij->oijl', dNdtheta, 1/self.dS) - \
            np.einsum('oij,lij,ij->oijl', dSdtheta, self.N, 1/(self.dS)**2)

        res['dndtheta'] = dndtheta
        res['dtildetheta'] = dtildetheta
        res['dSdtheta'] = dSdtheta

        return res
