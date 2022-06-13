from mpi4py import MPI
from src.surface.abstract_classes.abstract_surface import Surface
import numpy as np
import utilitiesRF as urf


def MGrid_from_surf(rmin, rmax, zmin, zmax, nfp, rad, zee, phi, myOutput, surf: Surface, j):
    """Generates a file that contains the magnetic field generated on a grid, by
    a current distribution carried by a surface. The format of the file is MGrid.

    :param rmin: minimum r of grid
    :type rmin: float

    :param rmax: maximum r of grid
    :type rmax: float

    :param zmin: minimum z of grid
    :type zmin: float

    :param zmax: maximum z of grid
    :type zmax: float

    :param nfp: number of field periods
    :type nfp: int

    :param rad: number of points in r direction
    :type rad: int

    :param zee: number of points in z direction
    :type zee: int

    :param phi: number of points in phi direction
    :type phi: int

    :param myOutput: name of the output file
    :type myOutput: str

    :param surf: surface that carries the current
    :type surf: Surface

    :param j: surface current distribution
    :type j: 3D float array

    :return: None
    :rtype: NoneType
    """
    from src.tools import get_rot_tensor, eijk
    from opt_einsum import contract
    from scipy.constants import mu_0

    print("Generating grid...")
    rs = np.linspace(rmin, rmax, rad)
    zs = np.linspace(zmin, zmax, zee)
    phis = np.linspace(0.0, np.pi / nfp, phi)
    PZRgrid = np.zeros((phi, zee, rad, 3))
    PZRgrid[:, :, :, 0], PZRgrid[:, :, :, 1], PZRgrid[:,
                                                      :, :, 2] = np.meshgrid(phis, zs, rs, indexing='ij')

    XYZgrid = np.empty(PZRgrid.shape)
    XYZgrid[..., 0] = PZRgrid[..., 2] * np.cos(PZRgrid[..., 0])
    XYZgrid[..., 1] = PZRgrid[..., 2] * np.sin(PZRgrid[..., 0])
    XYZgrid[..., 2] = PZRgrid[..., 1]
    print("...generated grid.")

    print("Evaluating Biot-Savart...")
    XYZBs = np.empty((phi, zee, rad, 3))

    rot_tensor = get_rot_tensor(surf.n_fp)

    T = XYZgrid[np.newaxis, np.newaxis, np.newaxis, :, :, :, :] - \
        contract('opq,ijq->oijp', rot_tensor,
                 surf.P)[:, :, :, np.newaxis, np.newaxis, np.newaxis, :]

    K = T / (np.linalg.norm(T, axis=-1)**3)[..., np.newaxis]

    XYZBs = mu_0 / (4*np.pi) * contract("niq,uvq,nuvpzrj,ijc,uv->pzrc", rot_tensor,
                                        j, K, eijk, surf.dS) / surf.npts

    rot = np.einsum("ijp->pij", np.array([
        [-np.sin(phis), np.cos(phis), np.zeros(phi)],
        [np.zeros(phi), np.zeros(phi), np.ones(phi)],
        [np.cos(phis), np.sin(phis), np.zeros(phi)]
    ]))

    PZRBs = np.einsum("pck,pzrk->pzrc", rot, XYZBs)
    print("...finished BiotSavart.")

    print("Writing MGrid...")
    mgridio = urf.MGridIO()
    mgridio.stringsize = 30
    mgridio.external_coil_groups = 1
    mgridio.dim_00001 = 1
    mgridio.external_coils = 1
    mgridio.rad = rad
    mgridio.zee = zee
    mgridio.phi = phi
    mgridio.nfp = nfp
    mgridio.nextcur = 1
    mgridio.rmin = rmin
    mgridio.rmax = rmax
    mgridio.zmin = zmin
    mgridio.zmax = zmax
    mgridio.coil_group = ['Total']
    mgridio.mgrid_mode = 'R'
    mgridio.raw_coil_cur = [1.0]
    rep = urf.StructuredGridRepresentation()
    mgridio.RZPhiGrid = urf.Volume(rep, data={'Begin': [0.0, zmin, rmin], 'End': [
        np.pi/nfp, zmax, rmax], 'nCells': [phi-1, zee-1, rad-1]})
    mgridio.magFields = [urf.Field(
        mgridio.RZPhiGrid, rep, tensorOrder=1, tensorDimensions=[3], data={'Data': PZRBs})]
    if MPI.COMM_WORLD.Get_rank() == 0:
        mgridio.write(myOutput)
    print("...finished writing MGrid.")
