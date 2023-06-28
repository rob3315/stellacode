import logging

from stellacode import np

from ..abstract_classes.abstract_pwc_3 import PWC_Surface_3

PI = np.pi
float_type = np.float64


class Surface_PWC_Ell_Tri_3(PWC_Surface_3):
    """A class used to represent an piecewise cylindrical surface with 3 cylinders / field period.
    With the section being represented by Fourier coefficients.

    Parameters
    ----------

    n_fp : int
        Number of field periods.
    R0 : float
        Radius of the Stellarator.
    l_u : int
        Number of poloidal angles per field period. Has to be even, and to be a multiple of n_cyl.
    l_v : int
        Number of toroidal angles.
    param : array
        parameters of the surface. Fourier coefficients and angles.
    """

    def __new__(cls, n_fp, l_u, l_v, param):
        if l_v % 4 != 0:
            raise ValueError("l_v has to be a multiple of four")
        else:
            return super(Surface_PWC_Ell_Tri_3, cls).__new__(cls)

    def __init__(self, n_fp, l_u, l_v, param):
        self.__n_fp = n_fp
        self.l_u = l_u
        self.l_v = l_v
        self.__param = param

        self.__a = self.__param[0]
        self.__kappa = self.__param[1]  # ellipticity
        self.__delta = self.__param[2]  # triangularity
        self.__R0 = self.__param[-3]  # major radius
        self.__alpha = self.__param[-2]
        self.__beta = self.__param[-1]

        # Computing the attributes of the surface
        self.__compute_r()

        self.compute_points()

        # First order attributes
        self.__compute_r_prime()

        self.compute_first_derivatives()

    @classmethod
    def from_file(cls, path_surf, n_fp, n_pol, n_tor):
        """
        Creates a Surface_PWC object from a text file.
        """
        file_extension = path_surf.split(".")[-1]

        if file_extension == "txt":
            data = []
            with open(path_surf, "r") as f:
                next(f)
                for line in f:
                    data.append(str.split(line))

            param = np.asarray(data[0], dtype=float_type)

            return cls(n_fp, n_pol, n_tor, param)

        elif file_extension == "json":
            import json

            with open(path_surf, "r") as f:
                data = json.load(f)

            a = data["surface"]["a"]
            kappa = data["surface"]["kappa"]
            delta = data["surface"]["delta"]
            R0 = data["surface"]["R0"]
            alpha = data["surface"]["alpha"]
            beta = data["surface"]["beta"]

            param = np.array([a, kappa, delta, R0, alpha, beta])

            return cls(n_fp, n_pol, n_tor, param)

        else:
            raise (ValueError, f"File extension: {file_extension} is not supported.")

    def _get_n_fp(self):
        """
        Number of field periods.
        """
        return self.__n_fp

    n_fp = property(_get_n_fp)

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
        return self.l_u * self.l_v

    npts = property(_get_npts)

    def _get_nbpts(self):
        return (self.l_u, self.l_v)

    nbpts = property(_get_nbpts)

    def _get_grids(self):
        u, v = (
            np.linspace(0, 1, self.l_u, endpoint=False),
            (np.arange(self.l_v) + 0.5) / self.l_v,
        )
        ugrid, vgrid = np.meshgrid(u, v, indexing="ij")
        return ugrid, vgrid

    grids = property(_get_grids)

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
        self.__a = self.__param[0]
        self.__kappa = self.__param[1]
        self.__delta = self.__param[2]
        self.__R0 = self.__param[-3]
        self.__alpha = self.__param[-2]
        self.__beta = self.__param[-1]
        self.__compute_r()
        self.__compute_r_prime()
        self.compute_points()
        self.compute_first_derivatives()

    param = property(_get_param, _set_param)

    def __compute_r(self):
        """
        Source
        ------

        https://www.sciencedirect.com/science/article/pii/S2405844017315050
        """

        def r(theta):
            return self.__a * np.sqrt(
                (1 + self.__delta * np.cos(theta)) ** 2 * np.cos(theta) ** 2 + self.__kappa**2 * np.sin(theta) ** 2
            )

        self.__r = r

    def __compute_r_prime(self):
        def r_prime(theta):
            return (
                self.__a**2
                * np.sin(2 * theta)
                * (self.__kappa**2 - (1 + self.__delta * np.cos(theta)) * (1 + 2 * self.__delta * np.cos(theta)))
                / (2 * self.__r(theta))
            )

        self.__r_prime = r_prime

    def get_theta_pertubation(self, compute_curvature=True):
        """
        Compute the perturbations of a surface
        """
        l_u, l_v = self.nbpts
        l_v_first_cylinder = l_v // 4
        us = np.linspace(0, 1, l_u, endpoint=False)
        vs = (np.arange(-l_v + 3 * l_v_first_cylinder, l_v_first_cylinder) + 0.5) / l_v

        ugrid, vgrid = np.meshgrid(us, vs, indexing="ij")

        thetagrid = 2 * np.pi * ugrid
        phigrid = 2 * np.pi / self.n_fp * vgrid

        perturbation = np.empty((len(self.__param), *ugrid.shape, 3), dtype=float_type)

        # dx / da
        perturbation[0, :, :, 0] = (
            np.cos(phigrid) * np.cos(thetagrid) * self.__r(thetagrid) / (self.__a * np.sin(phigrid + self.__alpha))
        )
        # dy / da
        perturbation[0, :, :, 1] = np.tan(phigrid) * perturbation[0, :, :, 0]
        # dz / da
        perturbation[0, :, :, 2] = np.tan(self.__beta) * perturbation[0, :, :, 1] + np.sin(thetagrid) * self.__r(
            thetagrid
        ) / (self.__a * np.cos(self.__beta))

        # dx / dkappa
        perturbation[1, :, :, 0] = (
            self.__a**2
            * self.__kappa
            * np.sin(thetagrid) ** 2
            * np.cos(phigrid)
            * np.cos(thetagrid)
            / (self.__r(thetagrid) * np.sin(phigrid + self.__alpha))
        )
        # dy / dkappa
        perturbation[1, :, :, 1] = np.tan(phigrid) * perturbation[1, :, :, 0]
        # dz / dkappa
        perturbation[1, :, :, 2] = np.tan(self.__beta) * perturbation[1, :, :, 1] + np.sin(
            thetagrid
        ) ** 3 * self.__a**2 * self.__kappa / (self.__r(thetagrid) * np.cos(self.__beta))

        # dx / ddelta
        perturbation[2, :, :, 0] = (
            self.__a**2
            * np.cos(phigrid)
            * np.cos(thetagrid) ** 4
            * (1 + self.__delta * np.cos(thetagrid))
            / (self.__r(thetagrid) * np.sin(phigrid + self.__alpha))
        )
        # dy / ddelta
        perturbation[2, :, :, 1] = np.tan(phigrid) * perturbation[2, :, :, 0]
        # dz / ddelta
        perturbation[2, :, :, 2] = np.tan(self.__beta) * perturbation[2, :, :, 1] + self.__a**2 * np.sin(
            thetagrid
        ) * np.cos(thetagrid) ** 3 * (1 + self.__delta * np.cos(thetagrid)) / (
            self.__r(thetagrid) * np.cos(self.__beta)
        )

        # dx / dR0
        perturbation[-3, :, :, 0] = np.tan(self.alpha) / (np.tan(self.alpha) + np.tan(phigrid))
        # dy / dR0
        perturbation[-3, :, :, 1] = np.tan(phigrid) * perturbation[-3, :, :, 0]
        # dz / dR0
        perturbation[-3, :, :, 2] = np.tan(self.beta) * perturbation[-3, :, :, 1]

        # dx / dalpha
        perturbation[-2, :, :, 0] = self.R0 * (1 + np.tan(self.alpha) ** 2) * np.tan(phigrid) / (
            np.tan(self.alpha) + np.tan(phigrid)
        ) ** 2 - np.cos(phigrid) * np.cos(phigrid + self.alpha) / np.sin(phigrid + self.alpha) ** 2 * np.cos(
            thetagrid
        ) * self.r(
            thetagrid
        )
        # dy / dalpha
        perturbation[-2, :, :, 1] = np.tan(phigrid) * perturbation[-2, :, :, 0]
        # dz / dalpha
        perturbation[-2, :, :, 2] = np.tan(self.beta) * perturbation[-2, :, :, 1]

        # dx / dbeta
        perturbation[-1, :, :, 0] = 0
        # dy / dbeta
        perturbation[-1, :, :, 1] = 0
        # dz / dbeta
        perturbation[-1, :, :, 2] = (1 + np.tan(self.beta) ** 2) * self.first_cylinder[::, ::, 1] + np.sin(
            self.beta
        ) / np.cos(self.beta) ** 2 * np.sin(thetagrid) * self.r(thetagrid)

        dperturbation = np.empty((len(self.__param), *ugrid.shape, 2, 3))

        # d²x / dphi da
        dperturbation[0, :, :, 1, 0] = (
            -np.cos(self.__alpha)
            * np.cos(thetagrid)
            * self.__r(thetagrid)
            / (self.__a * np.sin(phigrid + self.__alpha) ** 2)
        )
        # d²y / dphi da
        dperturbation[0, :, :, 1, 1] = (1 + np.tan(phigrid) ** 2) * perturbation[0, :, :, 0] + np.tan(
            phigrid
        ) * dperturbation[0, :, :, 1, 0]
        # d²z / dphi da
        dperturbation[0, :, :, 1, 2] = np.tan(self.beta) * dperturbation[0, :, :, 1, 1]

        # d²x / dtheta da
        dperturbation[0, :, :, 0, 0] = (
            np.cos(phigrid)
            * (self.__r_prime(thetagrid) * np.cos(thetagrid) - self.__r(thetagrid) * np.sin(thetagrid))
            / (self.__a * np.sin(phigrid + self.__alpha))
        )
        # d²y / dtheta da
        dperturbation[0, :, :, 0, 1] = np.tan(phigrid) * dperturbation[0, :, :, 0, 0]
        # d²z / dtheta da
        dperturbation[0, :, :, 0, 2] = np.tan(self.__beta) * dperturbation[0, :, :, 0, 1] + (
            self.__r(thetagrid) * np.cos(thetagrid) + self.__r_prime(thetagrid) * np.sin(thetagrid)
        ) / (self.__a * np.cos(self.__beta))

        # d²x / dphi dkappa
        dperturbation[1, :, :, 1, 0] = (
            -self.__a**2
            * np.cos(self.__alpha)
            * np.cos(thetagrid)
            * self.__kappa
            * np.sin(thetagrid) ** 2
            / (self.__r(thetagrid) * np.sin(phigrid + self.__alpha) ** 2)
        )
        # d²y / dphi dkappa
        dperturbation[1, :, :, 1, 1] = (1 + np.tan(phigrid) ** 2) * perturbation[1, :, :, 0] + np.tan(
            phigrid
        ) * dperturbation[1, :, :, 1, 0]
        # d²z / dphi dkappa
        dperturbation[1, :, :, 1, 2] = np.tan(self.beta) * dperturbation[1, :, :, 1, 1]

        # d²x / dtheta dkappa
        dperturbation[1, :, :, 0, 0] = (
            self.__a**2
            * self.__kappa
            * np.cos(phigrid)
            * np.sin(thetagrid)
            * (
                self.__r(thetagrid) * (3 * np.cos(thetagrid) ** 2 - 1)
                - np.cos(thetagrid) * np.sin(thetagrid) * self.__r_prime(thetagrid)
            )
            / (self.__r(thetagrid) ** 2 * np.sin(phigrid + self.__alpha))
        )
        # d²y / dtheta dkappa
        dperturbation[1, :, :, 0, 1] = np.tan(phigrid) * dperturbation[1, :, :, 0, 0]
        # d²z / dtheta dkappa
        dperturbation[1, :, :, 0, 2] = np.tan(self.__beta) * dperturbation[
            1, :, :, 0, 1
        ] + self.__a**2 * self.__kappa * np.sin(thetagrid) ** 2 * (
            3 * np.cos(thetagrid) * self.__r(thetagrid) - np.sin(thetagrid) * self.__r_prime(thetagrid)
        ) / (
            np.cos(self.__beta) * self.__r(thetagrid) ** 2
        )

        # d²x / dphi ddelta
        dperturbation[2, :, :, 1, 0] = (
            -np.cos(self.__alpha)
            * self.__a**2
            * np.cos(thetagrid) ** 4
            * (1 + self.__delta * np.cos(thetagrid))
            / (self.__r(thetagrid) * np.sin(phigrid + self.__alpha) ** 2)
        )
        # d²y / dphi ddelta
        dperturbation[2, :, :, 1, 1] = (1 + np.tan(phigrid) ** 2) * perturbation[2, :, :, 0] + np.tan(
            phigrid
        ) * dperturbation[2, :, :, 1, 0]
        # d²z / dphi ddelta
        dperturbation[2, :, :, 1, 2] = np.tan(self.__beta) * dperturbation[2, :, :, 1, 1]

        # d²x / dtheta ddelta
        dperturbation[2, :, :, 0, 0] = (
            -self.__a**2
            * np.cos(phigrid)
            * (
                self.__r(thetagrid)
                * np.sin(thetagrid)
                * np.cos(thetagrid) ** 3
                * (4 + 5 * self.__delta * np.cos(thetagrid))
                + self.__r_prime(thetagrid) * np.cos(thetagrid) ** 4 * (1 + self.__delta * np.cos(thetagrid))
            )
            / (np.sin(phigrid + self.__alpha) * self.__r(thetagrid) ** 2)
        )
        # d²y / dtheta ddelta
        dperturbation[2, :, :, 0, 1] = np.tan(phigrid) * dperturbation[2, :, :, 0, 0]
        # d²z / dtheta ddelta
        dperturbation[2, :, :, 0, 2] = np.tan(self.__beta) * dperturbation[2, :, :, 0, 1] + self.__a**2 / (
            np.cos(self.__beta) * self.__r(thetagrid) ** 2
        ) * (
            self.__r(thetagrid)
            * (
                5 * self.__delta * np.cos(thetagrid) ** 5
                + 4 * np.cos(thetagrid) ** 4
                - 4 * self.__delta * np.cos(thetagrid) ** 3
                - 3 * np.cos(thetagrid) ** 2
            )
            - self.__r_prime(thetagrid)
            * np.sin(thetagrid)
            * np.cos(thetagrid) ** 3
            * (1 + self.__delta * np.cos(thetagrid))
        )

        # d²x / dphi dR0
        dperturbation[-3, :, :, 1, 0] = (
            -np.tan(self.alpha) * (1 + np.tan(phigrid) ** 2) / (np.tan(self.alpha) + np.tan(phigrid)) ** 2
        )
        # d²y / dphi dR0
        dperturbation[-3, :, :, 1, 1] = (1 + np.tan(phigrid) ** 2) * perturbation[-3, :, :, 0] + np.tan(
            phigrid
        ) * dperturbation[-3, :, :, 1, 0]
        # d²z / dphi dR0
        dperturbation[-3, :, :, 1, 2] = np.tan(self.beta) * dperturbation[-3, :, :, 1, 1]

        # d²x / dtheta dR0
        dperturbation[-3, :, :, 0, 0] = 0
        # d²y / dtheta dR0
        dperturbation[-3, :, :, 0, 1] = 0
        # d²z / dtheta dR0
        dperturbation[-3, :, :, 0, 2] = 0

        # d²x / dphi dalpha
        dperturbation[-2, :, :, 1, 0] = self.R0 * (1 + np.tan(self.alpha) ** 2) * (1 + np.tan(phigrid) ** 2) * (
            np.tan(self.alpha) - np.tan(phigrid)
        ) / (np.tan(self.alpha) + np.tan(phigrid)) ** 3 + (
            np.sin(2 * phigrid + self.alpha) * np.sin(phigrid + self.alpha)
            + 2 * np.cos(phigrid) * np.cos(phigrid + self.alpha) ** 2
        ) / np.sin(
            phigrid + self.alpha
        ) ** 3 * np.cos(
            thetagrid
        ) * self.r(
            thetagrid
        )
        # d²y / dphi dalpha
        dperturbation[-2, :, :, 1, 1] = (1 + np.tan(phigrid) ** 2) * perturbation[-2, :, :, 0] + np.tan(
            phigrid
        ) * dperturbation[-2, :, :, 1, 0]
        # d²z / dphi dalpha
        dperturbation[-2, :, :, 1, 2] = np.tan(self.beta) * dperturbation[-2, :, :, 1, 1]

        # d²x / dtheta dalpha
        dperturbation[-2, :, :, 0, 0] = (
            np.cos(phigrid)
            * np.cos(phigrid + self.alpha)
            / np.sin(phigrid + self.alpha) ** 2
            * (np.sin(thetagrid) * self.r(thetagrid) - np.cos(thetagrid) * self.r_prime(thetagrid))
        )
        # d²y / dtheta dalpha
        dperturbation[-2, :, :, 0, 1] = np.tan(phigrid) * dperturbation[-2, :, :, 0, 0]
        # d²z / dtheta dalpha
        dperturbation[-2, :, :, 0, 2] = np.tan(self.beta) * dperturbation[-2, :, :, 0, 1]

        # d²x / dphi dbeta
        dperturbation[-1, :, :, 1, 0] = 0
        # d²y / dphi dbeta
        dperturbation[-1, :, :, 1, 1] = 0
        # d²z / dphi dbeta
        dperturbation[-1, :, :, 1, 2] = (
            (1 + np.tan(self.beta) ** 2) * self.dpsi_first_cylinder[1, 1, ::, ::] * self.n_fp / (2 * PI)
        )

        # d²x / dtheta dbeta
        dperturbation[-1, :, :, 0, 0] = 0
        # d²y / dtheta dbeta
        dperturbation[-1, :, :, 0, 1] = 0
        # d²z / dtheta dbeta
        dperturbation[-1, :, :, 0, 2] = (1 + np.tan(self.beta) ** 2) * self.dpsi_first_cylinder[0, 1, ::, ::] / (
            2 * PI
        ) + np.sin(self.beta) / np.cos(self.beta) ** 2 * (
            np.cos(thetagrid) * self.r(thetagrid) + np.sin(thetagrid) * self.r_prime(thetagrid)
        )

        # Conversion to u and v
        dperturbation[:, :, :, 0, :] *= 2 * PI
        dperturbation[:, :, :, 1, :] *= 2 * PI / self.n_fp

        second_cylinder = np.copy(perturbation[::, ::, ::-1, ::])

        phi_1 = np.pi / (2 * self.n_fp)

        symmetry_matrix = np.array(
            [
                [np.cos(2 * phi_1), np.sin(2 * phi_1), 0],
                [np.sin(2 * phi_1), -np.cos(2 * phi_1), 0],
                [0, 0, 1],
            ],
            dtype=float_type,
        )

        np.einsum("ij,kuvj->kuvi", symmetry_matrix, second_cylinder, out=second_cylinder)

        third_cylinder = np.copy(second_cylinder[::, ::, ::-1, ::])

        phi_1 = 2 * np.pi / self.n_fp - phi_1

        symmetry_matrix = np.array(
            [
                [np.cos(2 * phi_1), np.sin(2 * phi_1), 0],
                [np.sin(2 * phi_1), -np.cos(2 * phi_1), 0],
                [0, 0, 1],
            ]
        )

        np.einsum("ij,kuvj->kuvi", symmetry_matrix, third_cylinder, out=third_cylinder)

        full_perturbation = np.concatenate(
            (
                perturbation[::, ::, -l_v_first_cylinder::, ::],
                second_cylinder,
                third_cylinder[::, ::, :l_v_first_cylinder:, ::],
            ),
            axis=2,
        )

        dperturbation_second_cylinder = np.copy(dperturbation[::, ::, ::-1])

        phi_1 = np.pi / (2 * self.n_fp)

        sym_mat_1T = np.array(
            [
                [np.cos(2 * phi_1), np.sin(2 * phi_1), 0],
                [np.sin(2 * phi_1), -np.cos(2 * phi_1), 0],
                [0, 0, 1],
            ],
            dtype=float_type,
        )

        sym_mat_2T = np.array([[1, 0], [0, -1]], dtype=float_type)

        np.einsum(
            "ik,luvkj->luvij",
            sym_mat_2T,
            dperturbation_second_cylinder,
            out=dperturbation_second_cylinder,
        )
        np.einsum(
            "luvik,kj->luvij",
            dperturbation_second_cylinder,
            sym_mat_1T,
            out=dperturbation_second_cylinder,
        )

        dperturbation_third_cylinder = np.copy(dperturbation_second_cylinder[::, ::, ::-1])

        phi_1 = 2 * np.pi / self.n_fp - phi_1

        sym_mat_1T = np.array(
            [
                [np.cos(2 * phi_1), np.sin(2 * phi_1), 0],
                [np.sin(2 * phi_1), -np.cos(2 * phi_1), 0],
                [0, 0, 1],
            ],
            dtype=float_type,
        )

        sym_mat_2T = np.array([[1, 0], [0, -1]], dtype=float_type)

        np.einsum(
            "ik,luvkj->luvij",
            sym_mat_2T,
            dperturbation_third_cylinder,
            out=dperturbation_third_cylinder,
        )
        np.einsum(
            "luvik,kj->luvij",
            dperturbation_third_cylinder,
            sym_mat_1T,
            out=dperturbation_third_cylinder,
        )

        full_dperturbation = np.concatenate(
            (
                dperturbation[::, ::, -l_v_first_cylinder::, ::, ::],
                dperturbation_second_cylinder,
                dperturbation_third_cylinder[::, ::, :l_v_first_cylinder:, ::, ::],
            ),
            axis=2,
        )

        res = {}
        res["theta"] = full_perturbation
        res["dtheta"] = full_dperturbation

        dtilde_psi = np.array([self.dpsi[0], self.dpsi[1], self.n])
        partial_x_partial_u = np.linalg.inv(np.einsum("ijkl->klij", dtilde_psi))
        partial_x_partial_u_cut = partial_x_partial_u[:, :, :, :2]
        dtildetheta = np.einsum("ijkl,oijlm->oijkm", partial_x_partial_u_cut, res["dtheta"])
        # for cross product
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        dNdtheta = np.einsum("dij,aije,def->aijf", self.dpsi[0], res["dtheta"][:, :, :, 1, :], eijk)
        dNdtheta -= np.einsum("dij,aije,def->aijf", self.dpsi[1], res["dtheta"][:, :, :, 0, :], eijk)
        dSdtheta = np.einsum("aijd,dij->aij", dNdtheta, self.N) / self.dS
        dndtheta = np.einsum("oijl,ij->oijl", dNdtheta, 1 / self.dS) - np.einsum(
            "oij,lij,ij->oijl", dSdtheta, self.N, 1 / (self.dS) ** 2
        )

        res["dndtheta"] = dndtheta
        res["dtildetheta"] = dtildetheta
        res["dSdtheta"] = dSdtheta

        return res
