import numpy as np
from .abstract_surface import Surface
import logging


class Surface_Fourier(Surface):
    """A class used to represent an toroidal surface with Fourier coefficients

    :param surface_parametrization: (m,n,Rmn,Zmn) 4 lists to parametrize the surface
    :type surface_parametrization: (int[],int[],float[],float[])
    :param nbpts: see :func:`.abstract_surface.Abstract_surface`
    :type nbpts: (int,int)
    :param Np: see `.abstract_surface.Abstract_surface`
    :type Np: int
    """

    def __init__(self, surface_parametrization, nbpts, Np):
        self.Np = Np
        self.nbpts = nbpts
        self.npts = nbpts[0]*nbpts[1]
        self.surface_parametrization = surface_parametrization
        self.compute_surface_attributes()  # computation of the surface attributes
        logging.debug('creation of a Toroidal surface successfull')

    def load_file(pathfile):
        """load file with the format m,n,Rmn,Zmn"""
        data = []
        with open(pathfile, 'r') as f:
            for line in f:
                data.append(str.split(line))
        adata = np.array(data, dtype='float64')
        m, n, Rmn, Zmn = adata[:, 0], adata[:, 1], adata[:, 2], adata[:, 3]
        logging.debug('file extraction successfull')
        return (m, n, Rmn, Zmn)

    def change_param(param, dcoeff):
        """from a surface parameters and an array of modification,
        return the right surface parameters"""
        (m, n, Rmn, Zmn) = param
        dR = dcoeff[:len(m)]
        dZ = dcoeff[len(m):]
        return (m, n, Rmn+dR, Zmn+dZ)

    def compute_surface_attributes(self, deg=2):
        """compute surface elements used in the shape optimization up
        to degree deg
        deg is 0,1 or 2"""
        (m, n, Rmn, Zmn) = self.surface_parametrization
        lu, lv = self.nbpts  # nb of toroidal and poloidal point
        u, v = np.linspace(0, 1, lu, endpoint=False), np.linspace(
            0, 1, lv, endpoint=False)
        ugrid, vgrid = np.meshgrid(u, v, indexing='ij')
        R = np.zeros(ugrid.shape)
        Z = np.zeros(ugrid.shape)
        # first derivative
        if deg >= 1:
            dpsi = np.zeros((2, 3, lu, lv))
        # second derivative
        if deg >= 2:
            dpsi_uu = np.zeros((3, lu, lv))
            dpsi_uv = np.zeros((3, lu, lv))
            dpsi_vv = np.zeros((3, lu, lv))
            dRdu = np.zeros((lu, lv))
            dRdv = np.zeros((lu, lv))

        phi = 2*np.pi*vgrid/self.Np  # we draw only one segment of the whole torus
        # what we do (but faster):
        #        for i in range(len(u)):
        #            for j in range(len(v)):
        #                for k in range(len(n)):
        #                    R[i,j]+=Rmn[k]*np.cos(2*np.pi*(m[k]*u[i]-n[k]*v[j]))
        #                    Z[i,j]+=Zmn[k]*np.sin(2*np.pi*(m[k]*u[i]+n[k]*v[j]))
        # for sa in np.array_split(np.arange(len(m)), max(int(len(u)*len(v)*len(m)/Toroidal_surface.sat),1)): # to avoid memory saturation
        tmp = np.tensordot(m, ugrid, 0)+np.tensordot(n, vgrid, 0)  # m*u+n*v#
        # sum_n,m(Rmn*np.cos(2*pi*(m*u+n*v))
        R += np.tensordot(Rmn, np.cos(2*np.pi*tmp), 1)
        # sum_n,m(Zmn*np.sin(2*pi*(m*u+n*v))
        Z += np.tensordot(Zmn, np.sin(2*np.pi*tmp), 1)
        # first derivative
        if deg >= 1:
            # dR/du *cos(phi)=dX/du
            dpsi[0, 0, :, :] += np.tensordot(m*Rmn, -
                                             2*np.pi*np.sin(2*np.pi*tmp), 1)*np.cos(phi)
            # dR/du *sin(phi)=dY/du
            dpsi[0, 1, :, :] += np.tensordot(m*Rmn, -
                                             2*np.pi*np.sin(2*np.pi*tmp), 1)*np.sin(phi)
            # dZ/du
            dpsi[0, 2, :, :] += np.tensordot(m *
                                             Zmn, 2*np.pi*np.cos(2*np.pi*tmp), 1)
            # dR/dv *cos(phi)
            dpsi[1, 0, :, :] += np.tensordot(n*Rmn, -
                                             2*np.pi*np.sin(2*np.pi*tmp), 1)*np.cos(phi)
            # dR/dv *sin(phi)
            dpsi[1, 1, :, :] += np.tensordot(n*Rmn, -
                                             2*np.pi*np.sin(2*np.pi*tmp), 1)*np.sin(phi)
            # dZ/dv
            dpsi[1, 2, :, :] += np.tensordot(n *
                                             Zmn, 2*np.pi*np.cos(2*np.pi*tmp), 1)
        if deg >= 2:
            # second derivative
            # d^2R/du^2 *cos(phi)=dX/du
            dpsi_uu[0, :, :] += np.tensordot(m**2*Rmn, -
                                             (2*np.pi)**2*np.cos(2*np.pi*tmp), 1)*np.cos(phi)
            # d^2R/du^2 *sin(phi)=dY/du
            dpsi_uu[1, :, :] += np.tensordot(m**2*Rmn, -
                                             (2*np.pi)**2*np.cos(2*np.pi*tmp), 1)*np.sin(phi)
            # d^2Z/du^2
            dpsi_uu[2, :, :] += np.tensordot(m**2 *
                                             Zmn, -(2*np.pi)**2*np.sin(2*np.pi*tmp), 1)
            # d^2R/dudv *cos(phi)
            dpsi_uv[0, :, :] += np.tensordot(m*n*Rmn, -(2*np.pi)
                                             ** 2*np.cos(2*np.pi*tmp), 1)*np.cos(phi)
            # d^2R/dudv *sin(phi)
            dpsi_uv[1, :, :] += np.tensordot(m*n*Rmn, -(2*np.pi)
                                             ** 2*np.cos(2*np.pi*tmp), 1)*np.sin(phi)
            # d^2Z/dudv
            dpsi_uv[2, :, :] += np.tensordot(m*n *
                                             Zmn, -(2*np.pi)**2*np.sin(2*np.pi*tmp), 1)
            # d^2R/dv^2 *cos(phi)=dX/du
            dpsi_vv[0, :, :] += np.tensordot(n**2*Rmn, -
                                             (2*np.pi)**2*np.cos(2*np.pi*tmp), 1)*np.cos(phi)
            # d^2R/dv^2 *sin(phi)=dY/du
            dpsi_vv[1, :, :] += np.tensordot(n**2*Rmn, -
                                             (2*np.pi)**2*np.cos(2*np.pi*tmp), 1)*np.sin(phi)
            # d^2Z/dv^2
            dpsi_vv[2, :, :] += np.tensordot(n**2 *
                                             Zmn, -(2*np.pi)**2*np.sin(2*np.pi*tmp), 1)
            # other stuff
            dRdu += np.tensordot(m*Rmn, -2*np.pi*np.sin(2*np.pi*tmp), 1)
            dRdv += np.tensordot(n*Rmn, -2*np.pi*np.sin(2*np.pi*tmp), 1)

        # we save the result
        self.grids = (ugrid, vgrid)
        self.R = R
        self.Z = Z
        # we generate X and Y
        self.X = R*np.cos(phi)
        self.Y = R*np.sin(phi)
        self.P = np.einsum('kij->ijk', np.array([self.X, self.Y, self.Z]))
        if deg >= 1:
            dpsi[1, 0, :, :] += -R*2*np.pi / \
                self.Np*np.sin(phi)  # R dcos(phi)/dv
            dpsi[1, 1, :, :] += R*2*np.pi/self.Np*np.cos(phi)  # R dsin(phi)/dv
            # we save the result
            self.dpsi = dpsi
        if deg >= 2:
            dpsi_uv[0, :, :] += -dRdu*2*np.pi / \
                self.Np*np.sin(phi)  # R dcos(phi)/dv
            dpsi_uv[1, :, :] += dRdu*2*np.pi / \
                self.Np*np.cos(phi)  # R dsin(phi)/dv
            dpsi_vv[0, :, :] += -R*(2*np.pi/self.Np)**2 * \
                np.cos(phi)  # R d^2cos(phi)/dv^2
            dpsi_vv[1, :, :] += -R*(2*np.pi/self.Np)**2 * \
                np.sin(phi)  # R d^2sin(phi)/dv^2
            dpsi_vv[0, :, :] += -2 * dRdv*2*np.pi / \
                self.Np*np.sin(phi)  # R dcos(phi)/dv
            dpsi_vv[1, :, :] += 2 * dRdv*2*np.pi / \
                self.Np*np.cos(phi)  # R dsin(phi)/dv
            # we save the result
            self.dpsi_uu = dpsi_uu
            self.dpsi_uv = dpsi_uv
            self.dpsi_vv = dpsi_vv

        # We also compute surface element dS and derivatives dS_u and dS_v:
        if deg >= 1:
            N = np.cross(dpsi[0], dpsi[1], 0, 0, 0)
            self.N = N
            self.dS = np.linalg.norm(N, axis=0)
            self.n = N/self.dS  # normal inward unit vector
        if deg >= 2:
            dNdu = np.cross(dpsi_uu, dpsi[1], 0, 0, 0) + \
                np.cross(dpsi[0], dpsi_uv, 0, 0, 0)
            dNdv = np.cross(dpsi_uv, dpsi[1], 0, 0, 0) + \
                np.cross(dpsi[0], dpsi_vv, 0, 0, 0)
            self.dS_u = np.sum(dNdu*N, axis=0)/self.dS
            self.dS_v = np.sum(dNdv*N, axis=0)/self.dS
            self.n_u = dNdu/self.dS-self.dS_u*N/(self.dS**2)
            self.n_v = dNdv/self.dS-self.dS_v*N/(self.dS**2)
        # curvature computation
            # curvature computations :
            # First fundamental form of the surface (E,F,G)
            E = np.einsum('lij,lij->ij', dpsi[0], dpsi[0])
            F = np.einsum('lij,lij->ij', dpsi[0], dpsi[1])
            G = np.einsum('lij,lij->ij', dpsi[1], dpsi[1])
            self.I = (E, F, G)
            #m=np.cross(dpsi[0],dpsi[1],axisa=0, axisb=0)
            #p=np.sqrt(np.einsum('ijl,ijl->ij', m, m))
            # n=m/p[:,:,np.newaxis]
            # Second fundamental of the surface (L,M,N)
            L = np.einsum('lij,lij->ij', dpsi_uu, self.n)  # e
            M = np.einsum('lij,lij->ij', dpsi_uv, self.n)  # f
            N = np.einsum('lij,lij->ij', dpsi_vv, self.n)  # g
            self.II = (L, M, N)
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
            self.principles = principles

    def plot_surface(self):
        """Plot the surface"""
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(self.X, self.Y, self.Z,
                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()

    def get_boldpsi(self):
        """old, we should eliminate"""
        lu, lv = self.nbpts
        boldpsi = np.zeros((2, 3, lu, lv))
        boldpsi[0, :, :, :] = self.dpsi[0]/self.dS[np.newaxis, :, :]
        boldpsi[1, :, :, :] = self.dpsi[1]/self.dS[np.newaxis, :, :]
        return boldpsi

    def get_theta_pertubation(self, compute_curvature=True):
        """return a dictionary with the shape derivative of several elements"""
        (m, n, Rmn, Zmn) = self.surface_parametrization
        (lu, lv) = self.nbpts
        ugrid, vgrid = self.grids
        Np = self.Np
        ls = len(m)  # half of the number of degree of freedom for the surface
        # convention is first R then Z
        result = {}
        theta = np.zeros((2*ls, lu, lv, 3))  # perturbation of the surface
        dtheta = np.zeros((2*ls, lu, lv, 2, 3))  # perturbation of the surface
        # perturbation of extended vector field
        dtildetheta = np.zeros((2*ls, lu, lv, 3, 3))
        # div_S_theta=np.zeros((2*ls,lu,lv))

        phi = 2*np.pi*vgrid/Np
        tmp = np.tensordot(m, ugrid, 0)+np.tensordot(n, vgrid, 0)  # m*u+n*v#
        R = np.cos(2*np.pi*tmp)
        Z = np.sin(2*np.pi*tmp)
        theta[:ls, :, :, 0] = R*np.cos(phi)
        theta[:ls, :, :, 1] = R*np.sin(phi)
        theta[ls:, :, :, 2] = Z
        # first derivative
        dtheta[:ls, :, :, 0, 0] = np.einsum(
            'i,ijk->ijk', m, -2*np.pi*np.sin(2*np.pi*tmp)*np.cos(phi))  # dR/du *cos(phi)=dX/du
        dtheta[:ls, :, :, 0, 1] = np.einsum(
            'i,ijk->ijk', m, -2*np.pi*np.sin(2*np.pi*tmp)*np.sin(phi))  # dR/du *sin(phi)=dY/du
        dtheta[ls:, :, :, 0, 2] = np.einsum(
            'i,ijk->ijk', m, 2*np.pi*np.cos(2*np.pi*tmp))  # dZ/du
        dtheta[:ls, :, :, 1, 0] = np.einsum(
            'i,ijk->ijk', n, -2*np.pi*np.sin(2*np.pi*tmp)*np.cos(phi))  # dR/dv *cos(phi)
        # R dcos(phi)/dv
        dtheta[:ls, :, :, 1,
               0] += np.einsum('ijk,jk->ijk', R, -2*np.pi/self.Np*np.sin(phi))
        dtheta[:ls, :, :, 1, 1] = np.einsum(
            'i,ijk->ijk', n, -2*np.pi*np.sin(2*np.pi*tmp)*np.sin(phi))  # dR/dv *sin(phi)
        # R dsin(phi)/dv
        dtheta[:ls, :, :, 1,
               1] += np.einsum('ijk,jk->ijk', R, 2*np.pi/self.Np*np.cos(phi))
        dtheta[ls:, :, :, 1, 2] = np.einsum(
            'i,ijk->ijk', n, 2*np.pi*np.cos(2*np.pi*tmp))  # dZ/dv
        if compute_curvature:
            d2theta_uu = np.zeros((2*ls, lu, lv, 3))
            d2theta_uv = np.zeros((2*ls, lu, lv, 3))
            d2theta_vv = np.zeros((2*ls, lu, lv, 3))
            dRdu = np.einsum('o,oij->oij', m, -2*np.pi*np.sin(2*np.pi*tmp))
            dRdv = np.einsum('o,oij->oij', n, -2*np.pi*np.sin(2*np.pi*tmp))

            d2theta_uu[:ls, :, :, 0] += np.einsum('o,oij->oij', m**2, -(2*np.pi)**2*np.cos(
                2*np.pi*tmp)*np.cos(phi))  # d^2R/du^2 *cos(phi)=d^2X/du^2
            d2theta_uu[:ls, :, :, 1] += np.einsum('o,oij->oij', m**2, -(2*np.pi)**2*np.cos(
                2*np.pi*tmp)*np.sin(phi))  # d^2R/du^2 *sin(phi)=d^2Y/du^2
            d2theta_uu[ls:, :, :, 2] = np.einsum(
                'o,oij->oij', m**2, -(2*np.pi)**2*np.sin(2*np.pi*tmp))  # d^2Z/du^2

            d2theta_uv[:ls, :, :, 0] += np.einsum('o,oij->oij', m*n, -(
                2*np.pi)**2*np.cos(2*np.pi*tmp)*np.cos(phi))  # d^2R/dudv *cos(phi)
            # dR/du dcos(phi)/dv
            d2theta_uv[:ls, :, :,
                       0] += np.einsum('oij,ij->oij', -dRdu, 2*np.pi/self.Np*np.sin(phi))
            d2theta_uv[:ls, :, :, 1] += np.einsum('o,oij->oij', m*n, -(
                2*np.pi)**2*np.cos(2*np.pi*tmp)*np.sin(phi))  # d^2R/dudv *sin(phi)
            # dR/du dsin(phi)/dv
            d2theta_uv[:ls, :, :,
                       1] += np.einsum('oij,ij->oij', dRdu, 2*np.pi/self.Np*np.cos(phi))
            # d^2Z/dudv
            d2theta_uv[ls:, :, :, 2] += np.einsum(
                'o,oij->oij', m*n, -(2*np.pi)**2*np.sin(2*np.pi*tmp))

            d2theta_vv[:ls, :, :, 0] += np.einsum('o,oij->oij', n**2, -(
                2*np.pi)**2*np.cos(2*np.pi*tmp)*np.cos(phi))  # d^2R/dv^2 *cos(phi)
            # R d^2cos(phi)/dv^2
            d2theta_vv[:ls, :, :, 0] += -R * \
                ((2*np.pi/self.Np)**2*np.cos(phi))[np.newaxis, :, :]
            # dR/dv dcos(phi)/dv
            d2theta_vv[:ls, :, :, 0] += np.einsum(
                'oij,ij->oij', -2*dRdv, 2*np.pi/self.Np*np.sin(phi))

            d2theta_vv[:ls, :, :, 1] += np.einsum('o,oij->oij', n**2, -(
                2*np.pi)**2*np.cos(2*np.pi*tmp)*np.sin(phi))  # d^2R/dv^2 *sin(phi)
            # R d^2sin(phi)/dv^2
            d2theta_vv[:ls, :, :, 1] += -R * \
                ((2*np.pi/self.Np)**2*np.sin(phi))[np.newaxis, :, :]
            # dR/du dsin(phi)/dv
            d2theta_vv[:ls, :, :,
                       1] += np.einsum('oij,ij->oij', 2*dRdv, 2*np.pi/self.Np*np.cos(phi))

            d2theta_vv[ls:, :, :, 2] = np.einsum(
                'o,oij->oij', n**2, -(2*np.pi)**2*np.sin(2*np.pi*tmp))  # d^2Z/dv^2
            # dpsi_vv[1,:,:]+= -R*(2*np.pi/self.Np)**2*np.sin(phi)#R d^2sin(phi)/dv^2
            # dpsi_vv[1,:,:]+= 2* dRdv*2*np.pi/self.Np*np.cos(phi)#R dsin(phi)/dv

            result['d2theta'] = (d2theta_uu, d2theta_uv, d2theta_vv)

        # we use a local chart around the surface and invert it
        dtilde_psi = np.array([self.dpsi[0], self.dpsi[1], self.n])
        partial_x_partial_u = np.linalg.inv(
            np.einsum('ijkl->klij', dtilde_psi))  # we move axis and invert
        # we eliminate the last component as \partial_\delta \tilde \theta =0
        partial_x_partial_u_cut = partial_x_partial_u[:, :, :, :2]
        dtildetheta = np.einsum('ijkl,oijlm->oijkm',
                                partial_x_partial_u_cut, dtheta)
        # for cross product
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        dNdtheta = np.einsum('dij,aije,def->aijf',
                             self.dpsi[0], dtheta[:, :, :, 1, :], eijk)
        dNdtheta -= np.einsum('dij,aije,def->aijf',
                              self.dpsi[1], dtheta[:, :, :, 0, :], eijk)
        dSdtheta = np.einsum('aijd,dij->aij', dNdtheta, self.N)/self.dS
        dndtheta = np.einsum('oijl,ij->oijl', dNdtheta, 1/self.dS) - \
            np.einsum('oij,lij,ij->oijl', dSdtheta, self.N, 1/(self.dS)**2)
        result['dndtheta'] = dndtheta
        result['theta'] = theta
        result['dtildetheta'] = dtildetheta
        result['dtheta'] = dtheta
        result['dSdtheta'] = dSdtheta
        # derivation of the curvature
        # ndiv_S_theta += (self.dS_u+
        # print(np.max(np.einsum('ijklm,mjk,ljk->ijk',dtildetheta,self.n,self.n)))
        # TODO div_theta
        return result


def expand_for_plot(S):
    """from a toroidal_surface surface return X,Y,Z
    and add redundancy of first row"""
    shape = (S.X.shape[0]+1, S.X.shape[1])
    lst = []
    for elt in [S.X, S.Y, S.Z]:
        new_elt = np.zeros(shape)
        new_elt[:-1, :] = elt
        new_elt[-1, :] = elt[0, :]
        lst.append(new_elt.copy())
    return lst


def plot(lst_S):
    """plot a list of surfaces"""
    from mayavi import mlab
    lst_s = []
    for S in lst_S:
        X, Y, Z = expand_for_plot(S)
        lst_s.append(
            mlab.mesh(X, Y, Z, representation='surface', colormap='Wistia'))
    mlab.show()


def plot_function_on_surface(S, f):
    """Plot f the surface given by S.X,S.Y,S.Z"""
    from mayavi import mlab
    X, Y, Z = expand_for_plot(S)
    fc2 = np.concatenate((f, f[0:1, :]), axis=0)
    s = mlab.mesh(X, Y, Z, representation='surface', scalars=fc2)
    mlab.colorbar(s, nb_labels=4, label_fmt='%.1E', orientation='vertical')
    mlab.show()
    return(s)
