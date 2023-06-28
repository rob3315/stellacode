"""Various implementation of the main cost"""
# an example of Regcoil version in python
from dataclasses import dataclass
from scipy.constants import mu_0

import stellacode.tools as tools
import stellacode.tools.bnorm as bnorm
from stellacode import np


@dataclass
class EM_cost_dask_3:
    # Regularization parameter :
    lamb: float
    # Number of field periods :
    Np: int
    # Number of poloidal points on the cws :
    ntheta_coil: int
    # Number of toroidal points on the cws :
    nzeta_coil: int
    # Amount of current flowig poloidally :
    net_poloidal_current_Amperes: float
    # Amount of current flowig toroidally (usually 0) :
    net_toroidal_current_Amperes: float
    # De-normalization factor, found in the nescin file created when running STELLOPT BNORM :

    chunk_theta_plasma: int
    chunk_zeta_plasma: int
    phisize: tuple
    bnorm: np.array
    rot_tensor: np.array
    # matrixd_phi: np.array

    @classmethod
    def from_config(cls, config, Sp):
        mpol_coil = int(config["geometry"]["mpol_coil"])
        # Number of toroidal modes for the scalar current potential :
        ntor_coil = int(config["geometry"]["ntor_coil"])
        curpol = float(config["other"]["curpol"])
        BT = -curpol * bnorm.get_bnorm(str(config["other"]["path_bnorm"]), Sp)
        Np = int(config["geometry"]["Np"])
        # Tensor to compute the rotations
        rot_tensor = tools.get_rot_tensor(Np)
        phisize = (mpol_coil, ntor_coil)

        return cls(
            # Regularization parameter :
            lamb=float(config["other"]["lamb"]),
            # Number of field periods :
            Np=int(config["geometry"]["Np"]),
            # Number of poloidal points on the cws :
            ntheta_coil=int(config["geometry"]["ntheta_coil"]),
            # Number of toroidal points on the cws :
            nzeta_coil=int(config["geometry"]["nzeta_coil"]),
            # Amount of current flowig poloidally :
            net_poloidal_current_Amperes=float(config["other"]["net_poloidal_current_Amperes"]) / Np,
            # Amount of current flowig toroidally (usually 0) :
            net_toroidal_current_Amperes=float(config["other"]["net_toroidal_current_Amperes"]),
            chunk_theta_plasma=int(config["dask_parameters"]["chunk_theta_plasma"]),
            chunk_zeta_plasma=int(config["dask_parameters"]["chunk_zeta_plasma"]),
            phisize=phisize,
            bnorm=BT,
            rot_tensor=rot_tensor,
        )

    def get_cost(self, S, Sp, solve_cst_current: bool = True):
        BT = self.bnorm
        matrixd_phi = tools.get_matrix_dPhi(self.phisize, S.grids)

        dpsi = S.dpsi
        normalp = Sp.n
        S_dS = S.dS
        eijk = tools.eijk
        Qj = tools.compute_Qj(matrixd_phi, dpsi, S_dS)

        def compute_B(XYZgrid):
            T = (
                XYZgrid[np.newaxis, np.newaxis, np.newaxis, ...]
                - np.einsum("opq,ijq->oijp", self.rot_tensor, S.P)[..., np.newaxis, np.newaxis, :]
            )

            K = T / (np.linalg.norm(T, axis=-1) ** 3)[..., np.newaxis]

            B = (
                mu_0
                / (4 * np.pi)
                * np.einsum(
                    "sijpqa,tijh,sbc,hcij,dab,dpq->tpq",
                    K,
                    matrixd_phi,
                    self.rot_tensor,
                    dpsi,
                    eijk,
                    normalp,
                )
                / (self.ntheta_coil * self.nzeta_coil)
            )
            return B

        LS = compute_B(Sp.P)

        # Now we have to compute the best current components.
        # This is a technical part, one should read the paper :
        # "Optimal shape of stellarators for magnetic confinement fusion"
        # in order to understand what's going on.
        if solve_cst_current:
            LS_R = LS[2:]
            Qj_inv_R = np.linalg.inv(Qj[2:, 2:])
            B_tilde = BT - np.einsum(
                "tpq,t",
                LS[:2],
                [self.net_poloidal_current_Amperes, self.net_toroidal_current_Amperes],
            )
        else:
            LS_R = LS
            Qj_inv_R = np.linalg.inv(Qj)
            B_tilde = BT

        LS_dagger_R = np.einsum("ut,tij,ij->uij", Qj_inv_R, LS_R, Sp.dS / Sp.npts)

        # def solve_lambdas(self, LS_R, LS_dagger_R, B_tilde, Qj_inv_R, Qj, matrixd_phi, S, Sp,LS, BT):
        inside_M_lambda_R = self.lamb * np.eye(LS_R.shape[0]) + np.einsum("tpq,upq->tu", LS_dagger_R, LS_R)
        M_lambda_R = np.linalg.inv(inside_M_lambda_R)
        LS_dagger_B_tilde = np.einsum("hpq,pq->h", LS_dagger_R, B_tilde)

        if solve_cst_current:
            RHS = LS_dagger_B_tilde - self.lamb * Qj_inv_R @ (
                Qj[2:, :2]
                @ np.array(
                    [
                        self.net_poloidal_current_Amperes,
                        self.net_toroidal_current_Amperes,
                    ]
                )
            )
        else:
            RHS = LS_dagger_B_tilde

        j_S_R = M_lambda_R @ RHS

        if solve_cst_current:
            j_S = np.concatenate(
                (
                    np.array(
                        [
                            self.net_poloidal_current_Amperes,
                            self.net_toroidal_current_Amperes,
                        ]
                    ),
                    j_S_R,
                )
            )
        else:
            j_S = j_S_R

        # j_S is a vector containing the components of the best scalar current potential.
        # The real surface current is given by :
        j_3D = np.einsum("oijk,kdij,ij,o->ijd", matrixd_phi, S.dpsi, 1 / S.dS, j_S, optimize=True)

        # Save the results in a dictionnary :
        EM_cost_output = {}
        B_err = np.einsum("hpq,h", LS, j_S) - BT
        EM_cost_output["err_max_B"] = np.max(np.abs(B_err))
        EM_cost_output["max_j"] = np.max(np.linalg.norm(j_3D, axis=2))
        EM_cost_output["cost_B"] = self.Np * np.einsum("pq,pq,pq->", B_err, B_err, Sp.dS / Sp.npts)
        EM_cost_output["cost_J"] = self.Np * np.einsum("i,ij,j->", j_S, Qj, j_S)
        EM_cost_output["cost"] = EM_cost_output["cost_B"] + self.lamb * EM_cost_output["cost_J"]
        EM_cost_output["j_3D"] = j_3D
        EM_cost_output["j_S"] = j_S

        return EM_cost_output
