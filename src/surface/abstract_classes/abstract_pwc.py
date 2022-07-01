import numpy as np
from abc import abstractmethod

from .abstract_surface import Surface

float_type = np.float64
PI = np.pi


class PWC_Surface(Surface):
    """A class used to represent an abstract piecewise cylindrical toroidal surface.
    A PWC surface can be described by :
    - a cross-section, r : theta -> r(theta)
    - a major radius, R0
    - an angle inside the xy plane, alpha
    - an angle outside the xy plane, beta
    - a number of cylinders, field periods
    - Stellarator symmetry

    If those elements are provided, in a concrete child class of the PWC_Surface class,
    the current class can compute the positions and derivatives of the surface.
    The pwc_surfaces folder provides examples of concrete classes. 
    """
    @abstractmethod
    def _get_n_fp(self):
        """See Surface class.
        """
        pass

    n_fp = property(_get_n_fp)

    @abstractmethod
    def _get_n_cyl(self):
        """Gets the number of cylinders.

        :return: number of cylinders
        :rtype: int
        """
        pass

    n_cyl = property(_get_n_cyl)

    @abstractmethod
    def _get_symmetry(self):
        """Gets the Stellarator symmetry.

        :return: True iff the surface has Stellarator symmetry
        :rtype: bool
        """
        pass

    symmetry = property(_get_symmetry)

    @abstractmethod
    def _get_R0(self):
        """Gets the major radius.

        :return: major radius
        :rtype: float
        """
        pass

    R0 = property(_get_R0)

    @abstractmethod
    def _get_r(self):
        """Gets the function r(theta) which describes the section.

        :return: function r(theta)
        :rtype: callable
        """
        pass

    r = property(_get_r)

    @abstractmethod
    def _get_r_prime(self):
        """Gets the derivative of r(theta).

        :return: r'(theta)
        :rtype: callable
        """
        pass

    r_prime = property(_get_r_prime)

    @abstractmethod
    def _get_alpha(self):
        """Gets the angle alpha.
        Alpha is the angle in the xy plane.

        :return: alpha
        :rtype: float
        """
        pass

    alpha = property(_get_alpha)

    @abstractmethod
    def _get_beta(self):
        """Get the angle beta.
        Beta is the angle out of the xy plane.

        :return: beta
        :rtype: float
        """
        pass

    beta = property(_get_beta)

    def compute_points(self):
        """Computes the points of one part of the Stellarator.
        This function initializes the following attributes of the surface :
        - __P

        See Surface class for more information about this attribute.

        :return: None
        :rtype: NoneType
        """
        n_u, n_v = self.nbpts
        us = np.linspace(0, 1, n_u, endpoint=False, dtype=float_type)
        vs = (np.arange(n_v // self.n_cyl, dtype=float_type) + 0.5) / n_v
        ugrid, vgrid = np.meshgrid(us, vs, indexing='ij')

        thetagrid = 2 * PI * ugrid
        phigrid = 2 * PI / self.n_fp * vgrid

        points = np.empty((*ugrid.shape, 3), dtype=float_type)

        points[::, ::, 0] = self.R0 * np.tan(self.alpha) / (np.tan(self.alpha) + np.tan(phigrid)) + (
            np.cos(phigrid) / np.sin(phigrid + self.alpha)) * np.cos(thetagrid) * self.r(thetagrid)

        points[::, ::, 1] = np.tan(phigrid) * points[::, ::, 0]

        points[::, ::, 2] = np.tan(
            self.beta) * points[::, ::, 1] + np.sin(thetagrid) * self.r(thetagrid) / np.cos(self.beta)

        other_half = np.copy(points[::, ::-1, ::])

        angle = 2 * np.pi / (self.n_fp * self.n_cyl)

        symmetry_matrix = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        np.einsum("ij,uvj->uvi", symmetry_matrix,
                  other_half, out=other_half)

        two_cylinders = np.concatenate((points, other_half), axis=1)
        res = np.concatenate((points, other_half), axis=1)

        if self.symmetry:
            angle = 4 * np.pi / (self.n_fp * self.n_cyl)

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrix = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ], dtype=float_type)
                np.einsum("ij,uvj->uvi", symmetry_matrix,
                          two_cylinders[:, ::-1], out=two_cylinders)
                # np.einsum("ij,uvj->uvi", symmetry_matrix,
                # np.roll(two_cylinders[::-1, ::-1], 1, axis=0), out=two_cylinders)
                res = np.concatenate((res, two_cylinders), axis=1)
                angle += 4 * np.pi / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * np.pi / (self.n_fp * self.n_cyl)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=float_type)
            for _ in range(self.n_cyl // 2 - 1):
                np.einsum("ij,uvj->uvi", rotation_matrix,
                          two_cylinders, out=two_cylinders)
                res = np.concatenate((res, two_cylinders), axis=1)

        self.__P = res

    def _get_P(self):
        """Gets the points of the surface.

        :return: points
        :rtype: 3D array
        """
        return self.__P

    P = property(_get_P)

    def compute_first_derivatives(self):
        """Computes the derivatives of the transformation from the abstract torus to the real one.
        This transformation is a function psi : u, v -> x, y, z
        This function initializes the following attributes :
        - __dpsi
        - __N
        - __dS
        - __n

        See Surface class for more information about these attributes.

        :return: None
        :rtype: NoneType
        """
        n_u, n_v = self.nbpts
        us = np.linspace(0, 1, n_u, endpoint=False, dtype=float_type)
        vs = (np.arange(n_v // self.n_cyl, dtype=float_type) + 0.5) / n_v
        ugrid, vgrid = np.meshgrid(us, vs, indexing='ij')

        thetagrid = 2 * PI * ugrid
        phigrid = 2 * PI / self.n_fp * vgrid

        # Compute all partial derivatives d(x,y,z)/d(u, v)
        dxdtheta = (np.cos(phigrid) / np.sin(phigrid + self.alpha)) * (
            np.cos(thetagrid) * self.r_prime(thetagrid) - np.sin(thetagrid) * self.r(thetagrid))

        dxdu = dxdtheta * 2 * PI

        dxdphi = - self.R0 * np.tan(self.alpha) * (1 + np.tan(phigrid)**2) / (
            np.tan(self.alpha) + np.tan(phigrid))**2 - np.cos(thetagrid) * self.r(thetagrid) * np.cos(self.alpha) / np.sin(phigrid + self.alpha)**2

        dxdv = dxdphi * 2 * PI / self.n_fp

        dydtheta = np.tan(phigrid) * dxdtheta

        dydu = dydtheta * 2 * PI

        dydphi = (1 + np.tan(phigrid)**2) * \
            self.__P[::, :self.n_v // self.n_cyl:, 0] + \
            np.tan(phigrid) * dxdphi

        dydv = dydphi * 2 * PI / self.n_fp

        dzdtheta = np.tan(self.beta) * dydtheta + (np.cos(thetagrid) * self.r(
            thetagrid) + np.sin(thetagrid) * self.r_prime(thetagrid)) / np.cos(self.beta)

        dzdu = dzdtheta * 2 * PI

        dzdphi = np.tan(self.beta) * dydphi

        dzdv = dzdphi * 2 * PI / self.n_fp

        # Transpose of the Jacobian matrix of psi
        jacobianT = np.array([
            [dxdu, dydu, dzdu],
            [dxdv, dydv, dzdv]
        ])

        # Build the transposed Jacobian of the second cylinder
        # which is the symmetrical of the first one
        jacobianT_sym = np.copy(jacobianT[:, :, :, ::-1])

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

        np.einsum("ik,kjuv->ijuv", sym_mat_2T,
                  jacobianT_sym, out=jacobianT_sym)
        np.einsum("ikuv,kj->ijuv", jacobianT_sym,
                  sym_mat_1T, out=jacobianT_sym)

        # Transposed Jacobian of the first two cylinders
        two_cylinders_jacobianT = np.concatenate(
            (jacobianT, jacobianT_sym), axis=3)

        # Build transposed Jacobian of full field period
        full_jacobianT = np.concatenate((jacobianT, jacobianT_sym), axis=3)

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            sym_mat_2T = np.array([
                [1, 0],
                [0, -1]
            ], dtype=float_type)

            for _ in range(self.n_cyl // 2 - 1):
                symmetry_matrixT = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ], dtype=float_type)

                np.einsum("ik,kjuv->ijuv", sym_mat_2T,
                          two_cylinders_jacobianT[..., ::-1], out=two_cylinders_jacobianT)
                np.einsum("ikuv,kj->ijuv", two_cylinders_jacobianT,
                          symmetry_matrixT, out=two_cylinders_jacobianT)

                full_jacobianT = np.concatenate(
                    (full_jacobianT, two_cylinders_jacobianT), axis=3)

                angle += 4 * PI / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * PI / (self.n_fp * self.n_cyl)
            rotation_matrixT = np.array([
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=float_type)
            for _ in range(self.n_cyl // 2 - 1):
                np.einsum("ikuv,kj->uvij",
                          two_cylinders_jacobianT, rotation_matrixT, out=two_cylinders_jacobianT)

                full_jacobianT = np.concatenate(
                    (full_jacobianT, two_cylinders_jacobianT), axis=3)

        self.__dpsi = full_jacobianT

        self.__N = np.cross(self.__dpsi[0], self.__dpsi[1], 0, 0, 0)
        self.__dS = np.linalg.norm(self.__N, axis=0)
        self.__n = self.__N / self.__dS

    def _get_dpsi(self):
        """Gets the transposed Jacobian of the transformation.

        :return: transposed Jacobian
        :rtype: 4D array
        """
        return self.__dpsi

    dpsi = property(_get_dpsi)

    def _get_N(self):
        """Gets the normal inward vectors.

        :return: normal inward vectors
        :rtype: 3D array
        """
        return self.__N

    N = property(_get_N)

    def _get_dS(self):
        """Gets the surface elements.

        :return: surface elements
        :rtype: 2D array
        """
        return self.__dS

    dS = property(_get_dS)

    def _get_n(self):
        """Gets the normalized normal inward vectors.

        :return: normalized normal inward vectors
        :rtype: 3D array
        """
        return self.__n

    n = property(_get_n)

    def expand_for_plot_part(self):
        """Returns X, Y, Z arrays of one field period, adding redundancy of first column.

        :return: X, Y, Z arrays
        :rtype: tuple(2D array, 2D array, 2D array)
        """
        shape = self.__P.shape[0] + 1, self.__P.shape[1]

        X, Y, Z = np.empty(shape), np.empty(shape), np.empty(shape)
        X[:-1:, ::] = self.__P[::, ::, 0]
        X[-1, ::] = self.__P[0, ::, 0]
        Y[:-1:, ::] = self.__P[::, ::, 1]
        Y[-1, ::] = self.__P[0, ::, 1]
        Z[:-1:, ::] = self.__P[::, ::, 2]
        Z[-1, ::] = self.__P[0, ::, 2]

        return X, Y, Z

    def expand_for_plot_whole(self):
        """Returns X, Y, Z arrays of the whole Stellarator.

        :return: X, Y, Z arrays
        :rtype: tuple(2D array, 2D array, 2D array)
        """
        X, Y, Z = self.expand_for_plot_part()

        for _ in range(self.n_fp - 1):
            self.__rotation(2*PI / self.n_fp)
            newX, newY, newZ = self.expand_for_plot_part()
            X = np.concatenate((X, newX), axis=1)
            Y = np.concatenate((Y, newY), axis=1)
            Z = np.concatenate((Z, newZ), axis=1)

        self.__rotation(2*PI / self.n_fp)

        return np.concatenate((X, X[:, 0][:, np.newaxis]), axis=1), np.concatenate((Y, Y[:, 0][:, np.newaxis]), axis=1), np.concatenate((Z, Z[:, 0][:, np.newaxis]), axis=1)

    def plot_whole_surface(self, representation='surface'):
        """Plots the whole surface.

        :return: None
        :rtype: NoneType
        """
        from mayavi import mlab
        mlab.mesh(*self.expand_for_plot_whole(),
                  representation=representation, colormap='Wistia')
        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(
            100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100),
                    np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100),
                    np.linspace(0, 10, 100), color=(0, 0, 1))
        mlab.show()

    def __rotation(self, angle):
        """Rotation around the z axis of all the points generated.

        :return: None
        :rtype: NoneType
        """
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        np.einsum("ij,uvj->uvi", rotation_matrix, self.__P, out=self.__P)
