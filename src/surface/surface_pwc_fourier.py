import numpy as np
from .abstract_pwc import PWC_Surface
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
    n_u : int
        Number of poloidal angles per field period. Has to be even, and to be a multiple of n_cyl.
    n_v : int
        Number of toroidal angles.
    param : array
            parameters of the surface. Fourier coefficients and angles.
    """

    def __new__(cls, n_fp, n_cyl, symmetry, n_u, n_v, param):
        if n_v % 2 == 1:
            raise ValueError("n_v has to be even")
        elif n_cyl != 1 and n_cyl % 2 == 1:
            raise ValueError("n_cyl has to be equal to 1 or be even")
        elif n_v % n_cyl != 0:
            raise ValueError("n_v has to be a multiple of n_cyl")
        elif symmetry and n_cyl % 4 != 0 and n_cyl != 1:
            raise ValueError(
                "If symmetry, n_cyl has to be a multiple of 4 or be equal to 1")
        else:
            return super(Surface_PWC_Fourier, cls).__new__(cls)

    def __init__(self, n_fp, n_cyl, symmetry, n_u, n_v, param):
        if n_cyl == 1:
            n_cyl = 2
            param[-2] = PI / 2 - 2 * PI / n_fp
            param[-1] = 0
            symmetry = False

        self.__n_fp = n_fp
        self.__n_cyl = n_cyl
        # self.n_parts = n_fp * n_cyl // 2
        self.__symmetry = symmetry
        self.n_u = n_u
        self.n_v = n_v
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

    @classmethod
    def load_file(cls, path_surf, n_fp, n_pol, n_tor):
        """
        Creates a Surface_PWC object from a text file.
        """
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

    def _get_alpha(self):
        return self.__alpha

    alpha = property(_get_alpha)

    def _get_beta(self):
        return self.__beta

    beta = property(_get_beta)

    def _get_npts(self):
        return self.n_u * self.n_v

    npts = property(_get_npts)

    def _get_nbpts(self):
        return (self.n_u, self.n_v)

    nbpts = property(_get_nbpts)

    def _get_grids(self):
        u, v = np.linspace(0, 1, self.n_u, endpoint=False), (np.arange(
            self.n_v) + 0.5) / self.n_v
        ugrid, vgrid = np.meshgrid(u, v, indexing='ij')
        return ugrid, vgrid

    grids = property(_get_grids)

    """
    def _get_P(self):
        return self.__P

    P = property(_get_P)

    def _get_dpsi(self):
        return self.__dpsi

    dpsi = property(_get_dpsi)

    def _get_dS(self):
        return self.__dS

    dS = property(_get_dS)

    def _get_n(self):
        return self.__n

    n = property(_get_n)
    """

    def _get_principles(self):
        pass

    principles = property(_get_principles)

    def _get_I(self):
        pass

    I = property(_get_I)

    def _get_dpsi_uu(self):
        pass

    dpsi_uu = property(_get_dpsi_uu)

    def _get_dpsi_uv(self):
        pass

    dpsi_uv = property(_get_dpsi_uv)

    def _get_dpsi_vv(self):
        pass

    dpsi_vv = property(_get_dpsi_vv)

    def _get_II(self):
        pass

    II = property(_get_II)

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
        self.compute_points()
        self.compute_first_derivatives()

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

    def get_theta_pertubation(self, compute_curvature=True):
        """
        Compute the perturbations of a surface
        """
        us = np.linspace(0, 1, self.n_u, endpoint=False, dtype=float_type)
        vs = (np.arange(self.n_v // self.n_cyl, dtype=float_type) + 0.5) / self.n_v
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
        perturbation[-1, :, :, 2] = (1 + np.tan(self.beta)**2) * self.P[::, :self.n_v // self.n_cyl:, 1] + np.sin(
            self.beta) / np.cos(self.beta)**2 * np.sin(thetagrid) * self.r(thetagrid)

        dperturbation = np.empty(
            (n_coeffs + 3, *ugrid.shape, 3, 2))

        for k in range(n):
            # derivatives / phi
            dperturbation[k, :, :, 0, 0] = - np.cos(self.alpha) / np.sin(
                phigrid + self.alpha)**2 * np.cos(thetagrid) * np.cos(k * thetagrid)
            dperturbation[k, :, :, 1, 0] = (
                1 + np.tan(phigrid)**2) * perturbation[k, :, :, 0] + np.tan(phigrid) * dperturbation[k, :, :, 0, 0]
            dperturbation[k, :, :, 2, 0] = np.tan(
                self.beta) * dperturbation[k, :, :, 1, 0]

            dperturbation[k + n, :, :, 0,
                          0] = np.tan(k * thetagrid) * dperturbation[k, :, :, 0, 0]
            dperturbation[k + n, :, :, 1,
                          0] = np.tan(k * thetagrid) * dperturbation[k, :, :, 1, 0]
            dperturbation[k + n, :, :, 2,
                          0] = np.tan(k * thetagrid) * dperturbation[k, :, :, 2, 0]

            # derivatives / theta
            dperturbation[k, :, :, 0, 1] = - np.cos(phigrid) / np.sin(phigrid + self.alpha) * (
                np.sin(thetagrid) * np.cos(k * thetagrid) + k * np.cos(thetagrid) * np.sin(k * thetagrid))
            dperturbation[k, :, :, 1, 1] = np.tan(
                phigrid) * dperturbation[k, :, :, 0, 1]
            dperturbation[k, :, :, 2, 1] = np.tan(self.beta) * dperturbation[k, :, :, 1, 1] + 1 / np.cos(
                self.beta) * (np.cos(thetagrid) * np.cos(k * thetagrid) - k * np.sin(thetagrid) * np.sin(k * thetagrid))

            dperturbation[k + n, :, :, 0, 1] = np.tan(k * thetagrid) * dperturbation[k, :, :, 0, 1] + k * (
                1 + np.tan(k * thetagrid)**2) * perturbation[k, :, :, 0]
            dperturbation[k + n, :, :, 1, 1] = np.tan(k * thetagrid) * dperturbation[k, :, :, 1, 1] + k * (
                1 + np.tan(k * thetagrid)**2) * perturbation[k, :, :, 1]
            dperturbation[k + n, :, :, 2, 1] = np.tan(k * thetagrid) * dperturbation[k, :, :, 2, 1] + k * (
                1 + np.tan(k * thetagrid)**2) * perturbation[k, :, :, 2]

        dperturbation[-3, :, :, 0, 0] = - np.tan(self.alpha) * (
            1 + np.tan(phigrid)**2) / (np.tan(self.alpha) + np.tan(phigrid))**2
        dperturbation[-3, :, :, 1, 0] = (1 + np.tan(phigrid)**2) * perturbation[-3,
                                                                                :, :, 0] + np.tan(phigrid) * dperturbation[-3, :, :, 0, 0]
        dperturbation[-3, :, :, 2,
                      0] = np.tan(self.beta) * dperturbation[-3, :, :, 1, 0]

        dperturbation[-3, :, :, 0, 1] = 0
        dperturbation[-3, :, :, 1, 1] = 0
        dperturbation[-3, :, :, 2, 1] = 0

        # d²x / dphi dalpha
        dperturbation[-2, :, :, 0, 0] = self.R0 * (1 + np.tan(self.alpha)**2) * (1 + np.tan(phigrid)**2) * (np.tan(self.alpha) - np.tan(phigrid)) / (np.tan(self.alpha) + np.tan(phigrid))**3 + (
            np.sin(2*phigrid + self.alpha) * np.sin(phigrid + self.alpha) + 2 * np.cos(phigrid) * np.cos(phigrid + self.alpha)**2) / np.sin(phigrid + self.alpha)**3 * np.cos(thetagrid) * self.r(thetagrid)
        # d²y / dphi dalpha
        dperturbation[-2, :, :, 1, 0] = (1 + np.tan(phigrid)**2) * perturbation[-2,
                                                                                :, :, 0] + np.tan(phigrid) * dperturbation[-2, :, :, 0, 0]
        # d²z / dphi dalpha
        dperturbation[-2, :, :, 2,
                      0] = np.tan(self.beta) * dperturbation[-2, :, :, 1, 0]

        # d²x / dtheta dalpha
        dperturbation[-2, :, :, 0, 1] = np.cos(phigrid) * np.cos(phigrid + self.alpha) / np.sin(
            phigrid + self.alpha)**2 * (np.sin(thetagrid) * self.r(thetagrid) - np.cos(thetagrid) * self.r_prime(thetagrid))
        # d²y / dtheta dalpha
        dperturbation[-2, :, :, 1,
                      1] = np.tan(phigrid) * dperturbation[-2, :, :, 0, 1]
        # d²z / dtheta dalpha
        dperturbation[-2, :, :, 2,
                      1] = np.tan(self.beta) * dperturbation[-2, :, :, 1, 1]

        # d²x / dphi dbeta
        dperturbation[-1, :, :, 0, 0] = 0
        # d²y / dphi dbeta
        dperturbation[-1, :, :, 1, 0] = 0
        # d²z / dphi dbeta
        dperturbation[-1, :, :, 2,
                      0] = (1 + np.tan(self.beta)**2) * self.dpsi[1, 1, ::, :self.n_v // self.n_cyl:] * self.n_fp / (2 * PI)

        # d²x / dtheta dbeta
        dperturbation[-1, :, :, 0, 1] = 0
        # d²y / dtheta dbeta
        dperturbation[-1, :, :, 1, 1] = 0
        # d²z / dtheta dbeta
        dperturbation[-1, :, :, 2, 1] = (1 + np.tan(self.beta)**2) * \
            self.dpsi[0, 1, ::, :self.n_v // self.n_cyl:] / (2 * PI) + np.sin(self.beta) / np.cos(self.beta)**2 * (
                np.cos(thetagrid) * self.r(thetagrid) + np.sin(thetagrid) * self.r_prime(thetagrid))

        # Conversion to u and v
        dperturbation[:, :, :, :, 0] *= 2 * PI / self.n_fp
        dperturbation[:, :, :, :, 1] *= 2 * PI

        other_half = np.copy(perturbation[::, ::, ::-1, ::])

        angle = 2 * np.pi / (self.n_fp * self.n_cyl)

        symmetry_matrix = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        np.einsum("ij,kuvj->kuvi", symmetry_matrix, other_half, out=other_half)
        # np.dot(other_half, symmetry_matrix.T, out=other_half)

        two_cylinders_perturbation = np.concatenate(
            (perturbation, other_half), axis=2)
        full_perturbation = np.concatenate((perturbation, other_half), axis=2)

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, -1]
                ], dtype=float_type)

                np.einsum("ij,kuvj->kuvi", symmetry_matrix,
                          np.roll(two_cylinders_perturbation[::, ::-1, ::-1], 1, axis=1), out=two_cylinders_perturbation)
                # np.dot(two_cylinders_perturbation[::, ::-1, ::-1], symmetry_matrix.T, out=two_cylinders_perturbation)
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
                # np.dot(two_cylinders_perturbation, rotation_matrix.T, out=two_cylinders_perturbation)
                full_perturbation = np.concatenate(
                    (full_perturbation, two_cylinders_perturbation), axis=2)

        dperturbation_sym = np.copy(dperturbation[::, ::, ::-1])

        angle = 2 * PI / (self.n_fp * self.n_cyl)

        sym_mat_1 = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        sym_mat_2 = np.array([
            [-1, 0],
            [0, 1]
        ], dtype=float_type)

        np.einsum("ik,luvkj->luvij", sym_mat_1,
                  dperturbation_sym, out=dperturbation_sym)
        np.einsum("luvik,kj->luvij", dperturbation_sym,
                  sym_mat_2, out=dperturbation_sym)
        # dperturbation_sym = sym_mat_1 @ dperturbation_sym @ sym_mat_2

        two_cylinders_dperturbation = np.concatenate(
            (dperturbation, dperturbation_sym), axis=2)

        full_dperturbation = np.concatenate(
            (dperturbation, dperturbation_sym), axis=2)

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            sym_mat_2 = np.array([
                [-1, 0],
                [0, -1]
            ])

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, -1]
                ])

                np.einsum("ik,luvkj->luvij", symmetry_matrix, np.roll(
                    two_cylinders_dperturbation[::-1, ::-1], 1, axis=0), out=two_cylinders_dperturbation)
                np.einsum("luvik,kj->luvij", two_cylinders_dperturbation,
                          sym_mat_2, out=two_cylinders_dperturbation)

                # two_cylinders_dperturbation = symmetry_matrix @ np.roll(
                #   two_cylinders_dperturbation[::, ::-1, ::-1], 1, axis=1) @ sym_mat_2

                full_dperturbation = np.concatenate(
                    (full_dperturbation, two_cylinders_dperturbation), axis=2)

                angle += 4 * PI / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * PI / (self.n_fp * self.n_cyl)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for _ in range(self.n_cyl // 2 - 1):
                np.einsum("ik,luvkj->luvij", rotation_matrix,
                          two_cylinders_dperturbation, out=two_cylinders_dperturbation)
                # two_cylinders_dperturbation = rotation_matrix @ two_cylinders_dperturbation

                full_dperturbation = np.concatenate(
                    (full_dperturbation, two_cylinders_dperturbation), axis=2)

        full_dperturbation = np.einsum('fuvij->fuvji', full_dperturbation)
        full_dperturbation = full_dperturbation[:, :, :, ::-1]

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

        """

        us = np.linspace(0, 1, self.n_u, endpoint=False, dtype=float_type)
        vs = (np.arange(self.n_v // self.n_cyl, dtype=float_type) + 0.5) / self.n_v

        n_coeffs = len(self.fourier_coeffs)
        n = n_coeffs // 2
        perturbation = np.empty(
            (n_coeffs + 2, self.n_v // self.n_cyl, self.n_u, 3), dtype=float_type)

        thetas = 2 * PI * us
        phis = 2 * PI / self.n_fp * vs
        phis = phis[:, np.newaxis]

        for k in range(n):
            perturbation[k, :, :, 0] = np.cos(
                phis) / np.sin(phis + self.alpha) * np.cos(thetas) * np.cos(k * thetas)
            perturbation[k, :, :, 1] = np.tan(phis) * perturbation[k, :, :, 0]
            perturbation[k, :, :, 2] = np.tan(
                self.beta) * perturbation[k, :, :, 1] + np.sin(thetas) * np.cos(k * thetas) / np.cos(self.beta)
            perturbation[k + n, :, :,
                         0] = np.tan(k * thetas) * perturbation[k, :, :, 0]
            perturbation[k + n, :, :,
                         1] = np.tan(k * thetas) * perturbation[k, :, :, 1]
            perturbation[k + n, :, :,
                         2] = np.tan(k * thetas) * perturbation[k, :, :, 2]

        perturbation[-2, :, :, 0] = self.R0 * (1 + np.tan(self.alpha)**2) * np.tan(
            phis) / (np.tan(self.alpha) + np.tan(phis))**2 - np.cos(phis) * np.cos(phis + self.alpha) / np.sin(phis + self.alpha)**2 * np.cos(thetas) * self.r(thetas)
        perturbation[-2, :, :, 1] = np.tan(phis) * perturbation[-2, :, :, 0]
        perturbation[-2, :, :,
                     2] = np.tan(self.beta) * perturbation[-2, :, :, 1]

        perturbation[-1, :, :, 0] = 0
        perturbation[-1, :, :, 1] = 0
        perturbation[-1, :, :, 2] = (1 + np.tan(self.beta)**2) * self.__P[:self.n_v // self.n_cyl:, ::, 1] + np.sin(
            self.beta) / np.cos(self.beta)**2 * np.sin(thetas) * self.r(thetas)

        dperturbation = np.empty(
            (n_coeffs + 2, self.n_v // self.n_cyl, self.n_u, 3, 2))

        for k in range(n):
            # derivatives / phi
            dperturbation[k, :, :, 0, 0] = - np.cos(self.alpha) / np.sin(
                phis + self.alpha)**2 * np.cos(thetas) * np.cos(k * thetas)
            dperturbation[k, :, :, 1, 0] = (
                1 + np.tan(phis)**2) * perturbation[k, :, :, 0] + np.tan(phis) * dperturbation[k, :, :, 0, 0]
            dperturbation[k, :, :, 2, 0] = np.tan(
                self.beta) * dperturbation[k, :, :, 1, 0]

            dperturbation[k + n, :, :, 0,
                          0] = np.tan(k * thetas) * dperturbation[k, :, :, 0, 0]
            dperturbation[k + n, :, :, 1,
                          0] = np.tan(k * thetas) * dperturbation[k, :, :, 1, 0]
            dperturbation[k + n, :, :, 2,
                          0] = np.tan(k * thetas) * dperturbation[k, :, :, 2, 0]

            # derivatives / theta
            dperturbation[k, :, :, 0, 1] = - np.cos(phis) / np.sin(phis + self.alpha) * (
                np.sin(thetas) * np.cos(k * thetas) + k * np.cos(thetas) * np.sin(k * thetas))
            dperturbation[k, :, :, 1, 1] = np.tan(
                phis) * dperturbation[k, :, :, 0, 1]
            dperturbation[k, :, :, 2, 1] = np.tan(self.beta) * dperturbation[k, :, :, 1, 1] + 1 / np.cos(
                self.beta) * (np.cos(thetas) * np.cos(k * thetas) - k * np.sin(thetas) * np.sin(k * thetas))

            dperturbation[k + n, :, :, 0, 1] = np.tan(k * thetas) * dperturbation[k, :, :, 0, 1] + (
                1 + np.tan(thetas)**2) * perturbation[k, :, :, 0]
            dperturbation[k + n, :, :, 1, 1] = np.tan(k * thetas) * dperturbation[k, :, :, 1, 1] + (
                1 + np.tan(thetas)**2) * perturbation[k, :, :, 1]
            dperturbation[k + n, :, :, 2, 1] = np.tan(k * thetas) * dperturbation[k, :, :, 2, 1] + (
                1 + np.tan(thetas)**2) * perturbation[k, :, :, 2]

        # d²x / dphi dalpha
        dperturbation[-2, :, :, 0, 0] = self.R0 * (1 + np.tan(self.alpha)**2) * (1 + np.tan(phis)**2) * (np.tan(self.alpha) - np.tan(phis)) / (np.tan(self.alpha) + np.tan(phis))**3 + (
            np.sin(2*phis + self.alpha) * np.sin(phis + self.alpha) + 2 * np.cos(phis) * np.cos(phis + self.alpha)**2) / np.sin(phis + self.alpha)**3 * np.cos(thetas) * self.r(thetas)
        # d²y / dphi dalpha
        dperturbation[-2, :, :, 1, 0] = (1 + np.tan(phis)**2) * perturbation[-2,
                                                                             :, :, 0] + np.tan(phis) * dperturbation[-2, :, :, 0, 0]
        # d²z / dphi dalpha
        dperturbation[-2, :, :, 2,
                      0] = np.tan(self.beta) * dperturbation[-2, :, :, 1, 0]

        # d²x / dtheta dalpha
        dperturbation[-2, :, :, 0, 1] = np.cos(phis) * np.cos(phis + self.alpha) / np.sin(
            phis + self.alpha)**2 * (np.sin(thetas) * self.r(thetas) - np.cos(thetas) * self.r_prime(thetas))
        # d²y / dtheta dalpha
        dperturbation[-2, :, :, 1,
                      1] = np.tan(phis) * dperturbation[-2, :, :, 0, 1]
        # d²z / dtheta dalpha
        dperturbation[-2, :, :, 2,
                      1] = np.tan(self.beta) * dperturbation[-2, :, :, 1, 1]

        # d²x / dphi dbeta
        dperturbation[-1, :, :, 0, 0] = 0
        # d²y / dphi dbeta
        dperturbation[-1, :, :, 1, 0] = 0
        # d²z / dphi dbeta
        dperturbation[-1, :, :, 2,
                      0] = (1 + np.tan(self.beta)**2) * self.__dpsi[0, 1, :self.n_v // self.n_cyl:] * self.n_fp / (2 * PI)

        # d²x / dtheta dbeta
        dperturbation[-1, :, :, 0, 1] = 0
        # d²y / dtheta dbeta
        dperturbation[-1, :, :, 1, 1] = 0
        # d²z / dtheta dbeta
        dperturbation[-1, :, :, 2, 1] = (1 + np.tan(self.beta)**2) * \
            self.__dpsi[1, 1, :self.n_v // self.n_cyl:] / (2 * PI) + np.sin(self.beta) / np.cos(self.beta)**2 * (
                np.cos(thetas) * self.r(thetas) + np.sin(thetas) * self.r_prime(thetas))

        # Conversion to u and v
        dperturbation[:, :, :, :, 0] *= 2 * PI / self.n_fp
        dperturbation[:, :, :, :, 1] *= 2 * PI

        other_half = np.copy(perturbation[::, ::-1, ::, ::])

        angle = 2 * np.pi / (self.n_fp * self.n_cyl)

        symmetry_matrix = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        np.dot(other_half, symmetry_matrix.T, out=other_half)

        two_cylinders_perturbation = np.concatenate(
            (perturbation, other_half), axis=1)
        full_perturbation = np.concatenate((perturbation, other_half), axis=1)

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, -1]
                ], dtype=float_type)

                np.dot(two_cylinders_perturbation[::, ::-1, ::-1],
                       symmetry_matrix.T, out=two_cylinders_perturbation)
                full_perturbation = np.concatenate(
                    (full_perturbation, two_cylinders_perturbation), axis=1)
                angle += 4 * PI / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * PI / (self.n_fp * self.n_cyl)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for _ in range(self.n_cyl // 2 - 1):
                np.dot(two_cylinders_perturbation, rotation_matrix.T,
                       out=two_cylinders_perturbation)
                full_perturbation = np.concatenate(
                    (full_perturbation, two_cylinders_perturbation), axis=1)

        dperturbation_sym = np.empty(
            (n_coeffs + 2, self.n_u // self.n_cyl, self.n_v, 3, 2))

        angle = 2 * PI / (self.n_fp * self.n_cyl)

        sym_mat_1 = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        sym_mat_2 = np.array([
            [-1, 0],
            [0, 1]
        ], dtype=float_type)

        dperturbation_sym = sym_mat_1 @ dperturbation_sym @ sym_mat_2

        two_cylinders_dperturbation = np.concatenate(
            (dperturbation, dperturbation_sym), axis=1)

        full_dperturbation = np.concatenate(
            (dperturbation, dperturbation_sym), axis=1)

        if self.symmetry:
            angle = PI / self.n_parts

            sym_mat_2 = np.array([
                [-1, 0],
                [0, -1]
            ])

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, -1]
                ])

                two_cylinders_dperturbation = symmetry_matrix @ two_cylinders_dperturbation[::,
                                                                                            ::-1, ::-1] @ sym_mat_2

                full_dperturbation = np.concatenate(
                    (full_dperturbation, two_cylinders_dperturbation), axis=1)

                angle += 2 * PI / self.n_parts

        else:
            angle = 2 * PI / self.n_parts
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for _ in range(self.n_cyl // 2 - 1):
                two_cylinders_dperturbation = rotation_matrix @ two_cylinders_dperturbation

                full_dperturbation = np.concatenate(
                    (full_dperturbation, two_cylinders_dperturbation), axis=1)

        full_dperturbation = np.einsum('fuvij->fuvji', full_dperturbation)

        res = {}
        res['theta'] = full_perturbation
        res['dtheta'] = full_dperturbation

        dtilde_psi = np.array([self.__dpsi[0], self.__dpsi[1], self.__n])
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
                             self.__dpsi[0], res['dtheta'][:, :, :, 1, :], eijk)
        dNdtheta -= np.einsum('dij,aije,def->aijf',
                              self.__dpsi[1], res['dtheta'][:, :, :, 0, :], eijk)
        dSdtheta = np.einsum('aijd,dij->aij', dNdtheta, self.__N)/self.__dS
        dndtheta = np.einsum('oijl,ij->oijl', dNdtheta, 1/self.__dS) - \
            np.einsum('oij,lij,ij->oijl', dSdtheta, self.__N, 1/(self.__dS)**2)

        res['dndtheta'] = dndtheta
        res['dtildetheta'] = dtildetheta
        res['dSdtheta'] = dSdtheta

        return res

        """
