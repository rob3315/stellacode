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
    def _get_r_second(self):
        """Gets the second derivative of r(theta).

        :return: r''(theta)
        :rtype: callable
        """
        pass

    r_second = property(_get_r_second)

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
        l_u, l_v = self.nbpts
        us = np.linspace(0, 1, l_u, endpoint=False, dtype=float_type)
        vs = (np.arange(l_v // self.n_cyl, dtype=float_type) + 0.5) / l_v
        ugrid, vgrid = np.meshgrid(us, vs, indexing='ij')

        thetagrid = 2 * PI * ugrid
        phigrid = 2 * PI / self.n_fp * vgrid

        points = np.empty((*ugrid.shape, 3), dtype=float_type)

        points[..., 0] = np.cos(phigrid) / np.sin(phigrid + self.alpha) * (
            self.R0 * np.sin(self.alpha) + np.cos(thetagrid) * self.r(thetagrid))

        points[..., 1] = np.tan(phigrid) * points[..., 0]

        points[..., 2] = np.tan(
            self.beta) * points[::, ::, 1] + np.sin(thetagrid) * self.r(thetagrid) / np.cos(self.beta)

        if self.n_cyl == 1:
            self.__P = points

        else:
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
        l_u, l_v = self.nbpts
        us = np.linspace(0, 1, l_u, endpoint=False, dtype=float_type)
        vs = (np.arange(l_v // self.n_cyl, dtype=float_type) + 0.5) / l_v
        ugrid, vgrid = np.meshgrid(us, vs, indexing='ij')

        thetagrid = 2 * PI * ugrid
        phigrid = 2 * PI / self.n_fp * vgrid

        # Compute all partial derivatives d(x,y,z)/d(u, v)
        dxdtheta = (np.cos(phigrid) / np.sin(phigrid + self.alpha)) * (
            np.cos(thetagrid) * self.r_prime(thetagrid) - np.sin(thetagrid) * self.r(thetagrid))

        dxdu = dxdtheta * 2 * PI

        dxdphi = - np.cos(self.alpha) / np.sin(phigrid + self.alpha)**2 * \
            (self.R0 * np.sin(self.alpha) + np.cos(thetagrid) * self.r(thetagrid))

        dxdv = dxdphi * 2 * PI / self.n_fp

        dydtheta = np.tan(phigrid) * dxdtheta

        dydu = dydtheta * 2 * PI

        dydphi = (1 + np.tan(phigrid)**2) * \
            self.__P[::, :self.l_v // self.n_cyl:, 0] + \
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

        if self.n_cyl == 1:
            self.__dpsi = jacobianT

        else:
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

    def compute_second_derivatives(self):
        """Computes the second derivatives of the transformation from the abstract torus to the real one.
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
        l_u, l_v = self.nbpts
        us = np.linspace(0, 1, l_u, endpoint=False, dtype=float_type)
        vs = (np.arange(l_v // self.n_cyl, dtype=float_type) + 0.5) / l_v
        ugrid, vgrid = np.meshgrid(us, vs, indexing='ij')

        thetagrid = 2 * PI * ugrid
        phigrid = 2 * PI / self.n_fp * vgrid

        # First cylinder

        d2xdtheta2 = np.cos(phigrid) / np.sin(phigrid + self.alpha) * (np.cos(thetagrid) * self.r_second(
            thetagrid) - 2 * np.sin(thetagrid) * self.r_prime(thetagrid) - np.cos(thetagrid) * self.r(thetagrid))

        d2xdu2 = d2xdtheta2 * (2 * PI)**2

        d2xdphi2 = 2 * np.cos(self.alpha) * (self.R0 * np.sin(self.alpha) + np.cos(thetagrid)
                                             * self.r(thetagrid)) / (np.tan(phigrid + self.alpha) * np.sin(phigrid + self.alpha)**2)

        d2xdv2 = d2xdphi2 * (2 * PI / self.n_fp)**2

        d2xdthetadphi = np.cos(self.alpha) / np.sin(phigrid + self.alpha)**2 * (np.sin(
            thetagrid) * self.r(thetagrid) - np.cos(thetagrid) * self.r_prime(thetagrid))

        d2xdudv = d2xdthetadphi * 2 * PI * 2 * PI / self.n_fp

        d2ydtheta2 = np.tan(phigrid) * d2xdtheta2

        d2ydu2 = d2ydtheta2 * (2 * PI)**2

        d2ydphi2 = np.tan(phigrid) * d2xdphi2 + 2 * (1 + np.tan(phigrid)**2) * self.dpsi[1, 0, :, :self.l_v // self.n_cyl:] / (
            2 * PI / self.n_fp) + 2 * np.tan(phigrid) * (1 + np.tan(phigrid)**2) * self.P[::, :self.l_v // self.n_cyl:, 0]

        d2ydv2 = d2ydphi2 * (2 * PI / self.n_fp)**2

        d2ydthetadphi = np.tan(phigrid) * d2xdthetadphi + (1 + np.tan(phigrid)**2) * \
            self.dpsi[0, 0, :, :self.l_v // self.n_cyl:] / (2 * PI)

        d2ydudv = d2ydthetadphi * 2 * PI * 2 * PI / self.n_fp

        d2zdtheta2 = np.tan(self.beta) * d2ydtheta2 + (np.sin(thetagrid) * self.r_second(thetagrid) + 2 * np.cos(
            thetagrid) * self.r_prime(thetagrid) - np.sin(thetagrid) * self.r(thetagrid)) / np.cos(self.beta)

        d2zdu2 = d2zdtheta2 * (2 * PI)**2

        d2zdphi2 = np.tan(self.beta) * d2ydphi2

        d2zdv2 = d2zdphi2 * (2 * PI / self.n_fp)**2

        d2zdthetadphi = np.tan(self.beta) * d2ydthetadphi

        d2zdudv = d2zdthetadphi * 2 * PI * 2 * PI / self.n_fp

        d2psi1du2 = np.stack((d2xdu2, d2ydu2, d2zdu2), axis=0)
        d2psi1dv2 = np.stack((d2xdv2, d2ydv2, d2zdv2), axis=0)
        d2psi1dudv = np.stack((d2xdudv, d2ydudv, d2zdudv), axis=0)

        if self.n_cyl == 1:
            pass

        # Second cylinder

        angle = 2 * PI / (self.n_fp * self.n_cyl)

        sym_mat = np.array([
            [np.cos(2 * angle), np.sin(2 * angle), 0],
            [np.sin(2 * angle), - np.cos(2 * angle), 0],
            [0, 0, 1]
        ], dtype=float_type)

        d2psi2du2 = np.einsum("ij,juv->iuv", sym_mat, d2psi1du2[..., ::-1])
        d2psi2dv2 = np.einsum("ij,juv->iuv", sym_mat, d2psi1dv2[..., ::-1])
        d2psi2dudv = - np.einsum("ij,juv->iuv", sym_mat, d2psi1dudv[..., ::-1])

        # First two cylinders

        d2Psi0du2 = np.concatenate((d2psi1du2, d2psi2du2), axis=2)
        d2Psi0dv2 = np.concatenate((d2psi1dv2, d2psi2dv2), axis=2)
        d2Psi0dudv = np.concatenate((d2psi1dudv, d2psi2dudv), axis=2)

        # Full result

        fulld2psidu2 = np.concatenate((d2psi1du2, d2psi2du2), axis=2)
        fulld2psidv2 = np.concatenate((d2psi1dv2, d2psi2dv2), axis=2)
        fulld2psidudv = np.concatenate((d2psi1dudv, d2psi2dudv), axis=2)

        # Other cylinders

        if self.symmetry:
            angle = 4 * PI / (self.n_fp * self.n_cyl)

            for _ in range(self.n_cyl // 2 - 1):
                sym_mat = np.array([
                    [np.cos(2 * angle), np.sin(2 * angle), 0],
                    [np.sin(2 * angle), - np.cos(2 * angle), 0],
                    [0, 0, 1]
                ], dtype=float_type)

                np.einsum("ij,juv->iuv", sym_mat,
                          d2Psi0du2[..., ::-1], out=d2Psi0du2)
                np.einsum("ij,juv->iuv", sym_mat,
                          d2Psi0dv2[..., ::-1], out=d2Psi0dv2)
                np.einsum("ij,juv->iuv", - sym_mat,
                          d2Psi0dudv[..., ::-1], out=d2Psi0dudv)

                fulld2psidu2 = np.concatenate(
                    (fulld2psidu2, d2Psi0du2), axis=2)
                fulld2psidv2 = np.concatenate(
                    (fulld2psidv2, d2Psi0dv2), axis=2)
                fulld2psidudv = np.concatenate(
                    (fulld2psidudv, d2Psi0dudv), axis=2)

                angle += 4 * PI / (self.n_fp * self.n_cyl)

        else:
            angle = 4 * PI / (self.n_fp * self.n_cyl)
            rot_mat = np.array([
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], dtype=float_type)

            for _ in range(self.n_cyl // 2 - 1):
                np.einsum("ij,juv->iuv", rot_mat, d2Psi0du2, out=d2Psi0du2)
                np.einsum("ij,juv->iuv", rot_mat, d2Psi0dv2, out=d2Psi0dv2)
                np.einsum("ij,juv->iuv", rot_mat, d2Psi0dudv, out=d2Psi0dudv)

                fulld2psidu2 = np.concatenate(
                    (fulld2psidu2, d2Psi0du2), axis=2)
                fulld2psidv2 = np.concatenate(
                    (fulld2psidv2, d2Psi0dv2), axis=2)
                fulld2psidudv = np.concatenate(
                    (fulld2psidudv, d2Psi0dudv), axis=2)

        # Saving second derivatives

        self.__dpsi_uu = fulld2psidu2
        self.__dpsi_uv = fulld2psidudv
        self.__dpsi_vv = fulld2psidv2

        dNdu = np.cross(self.__dpsi_uu, self.dpsi[1], 0, 0, 0) + \
            np.cross(self.dpsi[0], self.__dpsi_uv, 0, 0, 0)
        dNdv = np.cross(self.__dpsi_uv, self.dpsi[1], 0, 0, 0) + \
            np.cross(self.dpsi[0], self.__dpsi_vv, 0, 0, 0)

        self.dS_u = np.sum(dNdu*self.N, axis=0)/self.dS
        self.dS_v = np.sum(dNdv*self.N, axis=0)/self.dS
        self.n_u = dNdu/self.dS-self.dS_u*self.N/(self.dS**2)
        self.n_v = dNdv/self.dS-self.dS_v*self.N/(self.dS**2)

        # Curvature computations :

        # First fundamental form of the surface (E,F,G)
        E = np.einsum('lij,lij->ij', self.dpsi[0], self.dpsi[0])
        F = np.einsum('lij,lij->ij', self.dpsi[0], self.dpsi[1])
        G = np.einsum('lij,lij->ij', self.dpsi[1], self.dpsi[1])
        self.__I = (E, F, G)

        # Second fundamental of the surface (L,M,N)
        L = np.einsum('lij,lij->ij', self.__dpsi_uu, self.n)  # e
        M = np.einsum('lij,lij->ij', self.__dpsi_uv, self.n)  # f
        N = np.einsum('lij,lij->ij', self.__dpsi_vv, self.n)  # g
        self.__II = (L, M, N)

        # K = det(second fundamental) / det(first fundamental)
        # Gaussian Curvature
        K = (L*N-M**2)/(E*G-F**2)
        self.K = K

        # trace of (second fundamental)(first fundamental^-1)
        # Mean Curvature
        H = ((E*N + G*L - 2*F*M)/((E*G - F**2)))/2
        self.H = H
        Pmax = H + np.sqrt(H**2 - K)
        Pmin = H - np.sqrt(H**2 - K)
        principles = [Pmax, Pmin]
        self.__principles = principles

    def _get_dpsi_uu(self):
        return self.__dpsi_uu

    dpsi_uu = property(_get_dpsi_uu)

    def _get_dpsi_uv(self):
        return self.__dpsi_uv

    dpsi_uv = property(_get_dpsi_uv)

    def _get_dpsi_vv(self):
        return self.__dpsi_vv

    dpsi_vv = property(_get_dpsi_vv)

    def _get_principles(self):
        return self.__principles

    principles = property(_get_principles)

    def _get_I(self):
        return self.__I

    I = property(_get_I)

    def _get_II(self):
        return self.__II

    II = property(_get_II)
