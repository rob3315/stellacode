import numpy as np
from .abstract_surface import Surface
import logging
from mayavi import mlab

PI = np.pi


class Surface_PWC(Surface):
    """A class used to represent an piecewise cylindrical surface.

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
        Number of toroidal angles per field period. Has to be even, and to be a multiple of n_cyl.
    n_v : int
        Number of poloidal angles.
    fourier_coeffs : 2D float array
        Fourier decomposition of the cross section. Each line contains a_n and b_n.
    alpha : float
        Angle between one cylinder and the Ox axis. Has to belong to ]0, pi[
    beta : float
        Angle between one cylinder and the Oy axis.
    """

    def __new__(cls, n_fp, n_cyl, symmetry, R0, n_u, n_v, fourier_coeffs, alpha, beta):
        if n_u % 2 == 1:
            raise ValueError("n_u has to be even")
        elif n_cyl != 1 and n_cyl % 2 == 1:
            raise ValueError("n_cyl has to be equal to 1 or be even")
        elif n_u % n_cyl != 0:
            raise ValueError("n_u has to be a multiple of n_cyl")
        elif symmetry and n_cyl % 4 != 0 and n_cyl != 1:
            raise ValueError(
                "If symmetry, n_cyl has to be a multiple of 4 or be equal to 1")
        else:
            return super(Surface_PWC, cls).__new__(cls)

    def __init__(self, n_fp, n_cyl, symmetry, R0, n_u, n_v, fourier_coeffs, alpha, beta):
        """
        Alpha is the angle from the x axis to the axis of the cylinder.
        Beta is the angle from the z axis to the axis of the cylinder.
        """
        if n_cyl == 1:
            n_cyl = 2
            alpha = PI / 2
            beta = 0
            symmetry = False

        self.n_fp = n_fp
        self.n_cyl = n_cyl
        self.n_parts = n_fp * n_cyl // 2
        self.symmetry = symmetry
        self.R0 = R0
        self.n_u = n_u
        self.n_v = n_v
        self.alpha = alpha
        self.beta = beta

        # Computing the attributes of the surface
        self.r = None
        self.__compute_r(fourier_coeffs)

        self.__P = None
        self.__compute_points()

        # First order attributes
        self.r_prime = None
        self.__compute_r_prime(fourier_coeffs)

        self.__dpsi = None
        self.__n = None
        self.__dS = None
        self.__compute_first_derivatives()

    @classmethod
    def load_file(cls, pathfile):
        """
        Creates a Surface_PWC object from a text file.
        """
        data = []
        with open(pathfile, 'r') as f:
            next(f)
            for line in f:
                data.append(str.split(line))

        n_fp = int(data[0][0])
        n_cyl = int(data[0][1])
        symmetry = data[0][2] == "True"
        R0 = float(data[0][3])
        n_u = int(data[0][4])
        n_v = int(data[0][5])
        alpha = float(data[0][6])
        beta = float(data[0][7])

        data.pop(0)
        fourier_coeffs = np.array(data, dtype='float64')

        logging.debug('Fourier coefficients extracted from file')

        return cls(n_fp, n_cyl, symmetry, R0, n_u, n_v, fourier_coeffs, alpha, beta)

    def _get_npts(self):
        return self.n_u * self.n_v

    npts = property(_get_npts)

    def _get_nbpts(self):
        return (self.n_u, self.n_v)

    nbpts = property(_get_nbpts)

    def _get_grids(self):
        us = np.arange(0, self.n_u) / self.n_u
        vs = np.linspace(0, 1, self.n_v, endpoint=False)
        return np.meshgrid(us, vs, indexing='ij')

    grids = property(_get_grids)

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

    def change_param(param, dcoeff):
        pass

    def get_theta_pertubation(self, compute_curvature):
        """
        Compute the perturbations of a surface
        """
        pass

    def __compute_r(self, fourier_coeffs):
        """
        Returns a function which describes the cross section of the cylinder.

        Parameters
        ----------

        fourier_coeffs : array
            2D array. The n-th row has the two Fourier coefficients a_n and b_n.
        """
        def r(theta):
            n = len(fourier_coeffs)
            return np.sum(fourier_coeffs * np.column_stack((np.cos(theta * np.arange(n)), np.sin(theta * np.arange(n)))))
        self.r = np.vectorize(r)

    def __compute_r_prime(self, fourier_coeffs):
        def r_prime(theta):
            n = len(fourier_coeffs)
            return np.sum(fourier_coeffs[1::, ::] * np.column_stack((- np.arange(1, n) * np.sin(theta * np.arange(1, n)), np.arange(1, n) * np.cos(theta * np.arange(1, n)))))
        self.r_prime = np.vectorize(r_prime)

    def __compute_points(self):
        """
        Computes the points of one part of the Stellarator.
        """
        us = (2 * np.arange(self.n_u // self.n_cyl) + 1) / self.n_u
        vs = np.linspace(0, 1, self.n_v, endpoint=False)

        points = np.empty((self.n_u // self.n_cyl, self.n_v, 3))

        thetas = 2 * PI * vs
        phis = - 2 * PI / (self.n_fp * self.n_cyl) + PI / self.n_fp * us
        phis = phis[:, np.newaxis]

        points[::, ::, 0] = self.R0 * np.tan(self.alpha) / (np.tan(self.alpha) + np.tan(phis)) + (
            np.cos(phis) / np.sin(phis + self.alpha)) * np.cos(thetas) * self.r(thetas)

        points[::, ::, 1] = np.tan(phis) * points[::, ::, 0]

        points[::, ::, 2] = np.tan(
            self.beta) * points[::, ::, 1] + np.sin(thetas) * self.r(thetas) / np.cos(self.beta)

        other_half = np.copy(points[::-1, ::, ::])
        other_half[::, ::, 1] *= -1

        if self.symmetry:
            other_half[::, ::, 2] = -1 * other_half[::, ::-1, 2]

        two_cylinders = np.concatenate((points, other_half), axis=0)
        res = np.concatenate((points, other_half), axis=0)

        if self.symmetry:
            angle = PI / self.n_parts

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ])

                np.dot(two_cylinders[::-1],
                       symmetry_matrix.T, out=two_cylinders)
                res = np.concatenate((res, two_cylinders))
                angle += 2 * PI / self.n_parts

        else:
            angle = 2*PI / self.n_parts
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for _ in range(self.n_cyl // 2 - 1):
                np.dot(two_cylinders, rotation_matrix.T, out=two_cylinders)
                res = np.concatenate((res, two_cylinders))

        self.__P = res

    def __compute_first_derivatives(self):
        """
        Computes the derivatives of the transformation.
        """
        us = (2 * np.arange(self.n_u // self.n_cyl) + 1) / self.n_u
        vs = np.linspace(0, 1, self.n_v, endpoint=False)

        thetas = 2 * PI * vs
        phis = - 2 * PI / (self.n_fp * self.n_cyl) + PI / self.n_fp * us
        phis = phis[:, np.newaxis]

        dxdphi = np.empty((self.n_u // self.n_cyl, self.n_v))

        dxdphi = - self.R0 * np.tan(self.alpha) * (1 + np.tan(phis)**2) / (
            np.tan(self.alpha) + np.tan(phis))**2 - np.cos(thetas) * self.r(thetas) * np.cos(self.alpha) / np.sin(phis + self.alpha)**2

        dxdu = dxdphi * 2 * PI / self.n_parts

        dxdtheta = (np.cos(phis) / np.sin(phis + self.alpha)) * (
            np.cos(thetas) * self.r_prime(thetas) - np.sin(thetas) * self.r(thetas))

        dxdv = dxdtheta * 2 * PI

        dydphi = (1 + np.tan(phis)**2) * \
            self.__P[:self.n_u // self.n_cyl:,
                     ::, 0] + np.tan(phis) * dxdphi

        dydu = dydphi * 2 * PI / self.n_parts

        dydtheta = np.tan(phis) * dxdtheta

        dydv = dydtheta * 2 * PI

        dzdphi = np.tan(self.beta) * dydphi

        dzdu = dzdphi * 2 * PI / self.n_parts

        dzdtheta = np.tan(self.beta) * dydtheta + (np.cos(thetas) * self.r(thetas) +
                                                   np.sin(thetas) * self.r_prime(thetas)) / np.cos(self.beta)

        dzdv = dzdtheta * 2 * PI

        jacobian = np.einsum('ijuv->uvij', np.array([
            [dxdu, dxdv],
            [dydu, dydv],
            [dzdu, dzdv]
        ]))

        jacobian_sym = np.empty((self.n_u // self.n_cyl, self.n_v, 3, 2))

        if not self.symmetry:

            jacobian_sym[::, ::] = jacobian[::-1, ::]

            sym_mat_1 = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])

            sym_mat_2 = np.array([
                [-1, 0],
                [0, 1]
            ])

        else:

            jacobian_sym[::, ::] = jacobian[::-1, ::-1]

            sym_mat_1 = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])

            sym_mat_2 = np.array([
                [-1, 0],
                [0, -1]
            ])

        jacobian_sym = sym_mat_1 @ jacobian_sym @ sym_mat_2

        two_cylinders_jacobian = np.concatenate(
            (jacobian, jacobian_sym), axis=0)

        full_jacobian = np.concatenate((jacobian, jacobian_sym), axis=0)

        if self.symmetry:
            angle = PI / self.n_parts

            sym_mat_2 = np.array([
                [-1, 0],
                [0, 1]
            ])

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ])

                two_cylinders_jacobian = symmetry_matrix @ two_cylinders_jacobian[::-1] @ sym_mat_2

                full_jacobian = np.concatenate(
                    (full_jacobian, two_cylinders_jacobian))

                angle += 2 * PI / self.n_parts

        else:
            angle = 2 * PI / self.n_parts
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            for _ in range(self.n_cyl // 2 - 1):
                two_cylinders_jacobian = rotation_matrix @ two_cylinders_jacobian

                full_jacobian = np.concatenate(
                    (full_jacobian, two_cylinders_jacobian))

        self.__dpsi = np.einsum('uvij->jiuv', full_jacobian)

        normal = np.cross(self.__dpsi[1], self.__dpsi[0], 0, 0, 0)
        self.__dS = np.linalg.norm(normal, axis=0)
        self.__n = normal / self.__dS

    def __expand_for_plot_part(self):
        """
        from a toroidal_surface surface return X,Y,Z
        and add redundancy of first column
        """
        shape = self.__P.shape[0], self.__P.shape[1] + 1

        X, Y, Z = np.empty(shape), np.empty(shape), np.empty(shape)
        X[::, :-1:] = self.__P[::, ::, 0]
        X[::, -1] = self.__P[::, 0, 0]
        Y[::, :-1:] = self.__P[::, ::, 1]
        Y[::, -1] = self.__P[::, 0, 1]
        Z[::, :-1:] = self.__P[::, ::, 2]
        Z[::, -1] = self.__P[::, 0, 2]

        return X, Y, Z

    def __expand_for_plot_whole(self):
        """
        from a toroidal_surface surface return X,Y,Z
        and add redundancy of first column
        """
        X, Y, Z = self.__expand_for_plot_part()

        for _ in range(self.n_fp - 1):
            self.__rotation(2*PI / self.n_fp)
            newX, newY, newZ = self.__expand_for_plot_part()
            X = np.concatenate((X, newX), axis=0)
            Y = np.concatenate((Y, newY), axis=0)
            Z = np.concatenate((Z, newZ), axis=0)

        self.__rotation(2*PI / self.n_fp)

        return np.concatenate((X, X[0][np.newaxis, :]), axis=0), np.concatenate((Y, Y[0][np.newaxis, :]), axis=0), np.concatenate((Z, Z[0][np.newaxis, :]), axis=0)

    def plot_part(self, representation='surface'):
        mlab.mesh(*self.__expand_for_plot_part(),
                  representation=representation, colormap='Wistia')
        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(
            100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100),
                    np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100),
                    np.linspace(0, 10, 100), color=(0, 0, 1))
        mlab.show()

    def plot_whole_surface(self, representation='surface'):
        mlab.mesh(*self.__expand_for_plot_whole(),
                  representation=representation, colormap='Wistia')
        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(
            100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100),
                    np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100),
                    np.linspace(0, 10, 100), color=(0, 0, 1))
        mlab.show()

    def __rotation(self, angle):
        """
        Rotation around the z axis of all the points generated.
        """
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        np.dot(self.__P, rotation_matrix.T, out=self.__P)
