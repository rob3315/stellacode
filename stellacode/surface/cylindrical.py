from os import sep

from scipy.io import netcdf_file

from stellacode import np

from .abstract_surface import AbstractSurface


class FourierSurface(AbstractSurface):
    """A class used to represent an toroidal surface with Fourier coefficients

    :param params: (m,n,Rmn,Zmn) 4 lists to parametrize the surface
    :type params: (int[],int[],float[],float[])
    :param nbpts: see :func:`.abstract_surface.Abstract_surface`
    :type nbpts: (int,int)
    :param Np: see `.abstract_surface.Abstract_surface`
    :type Np: int
    """

    def __init__(self, params, nbpts, Np):
        self.Np = Np
        self.nbpts = nbpts
        self.npts = nbpts[0] * nbpts[1]
        self.params = params
        self.param = np.concatenate((params[2], params[3]))
        self.compute_surface_attributes()  # computation of the surface attributes

    @classmethod
    def from_file(cls, path_surf, n_fp, n_pol, n_tor):
        """This function returns a Surface_Fourier object defined by a file.
        Three kinds of file are currently supported :
        - wout_.nc files generated by VMEC (for plasma surfaces)
        - nescin files generated by regcoil
        - text files (see example in data/li383)

        load file with the format m,n,Rmn,Zmn"""

        if path_surf[-3::] == ".nc":
            f = netcdf_file(path_surf, "r", mmap=False)
            m = f.variables["xm"][()]
            n = -f.variables["xn"][()] / n_fp
            Rmn = f.variables["rmnc"][()][-1]
            Zmn = f.variables["zmns"][()][-1]
            f.close()

        elif path_surf.rpartition(sep)[-1][:6:] == "nescin":
            with open(path_surf, "r") as f:
                line = f.readline()
                while "crc" not in line:
                    line = f.readline()
                data = []
                for line in f:
                    data.append(str.split(line))
                adata = np.array(data, dtype="float64")
                m, n, Rmn, Zmn = adata[:, 0], adata[:, 1], adata[:, 2], adata[:, 3]
        else:
            data = []
            with open(path_surf, "r") as f:
                next(f)
                for line in f:
                    data.append(str.split(line))

            adata = np.array(data, dtype="float64")
            m, n, Rmn, Zmn = adata[:, 0], adata[:, 1], adata[:, 2], adata[:, 3]

        params = (m, n, Rmn, Zmn)

        return cls(params, (n_pol, n_tor), n_fp)

    def _get_param(self):
        return self.__param

    def _set_param(self, param):
        self.__param = param
        m, n = self.params[0], self.params[1]
        Rmn, Zmn = param[: len(m)], param[len(m) :]
        self.params = (m, n, Rmn, Zmn)
        self.compute_surface_attributes()

    param = property(_get_param, _set_param)

    def change_param(param, dcoeff):
        """from a surface parameters and an array of modification,
        return the right surface parameters"""
        (m, n, Rmn, Zmn) = param
        dR = dcoeff[: len(m)]
        dZ = dcoeff[len(m) :]
        return (m, n, Rmn + dR, Zmn + dZ)

    def get_xyz(self, uv):
        m, n, Rmn, Zmn = self.params
        u, v = uv
        tmp = u * m + v * n
        R = np.tensordot(Rmn, np.cos(2 * np.pi * tmp), 1)
        Z = np.tensordot(Zmn, np.sin(2 * np.pi * tmp), 1)
        phi = 2 * np.pi * v / self.Np
        return np.array([R * np.cos(phi), R * np.sin(phi), Z])
