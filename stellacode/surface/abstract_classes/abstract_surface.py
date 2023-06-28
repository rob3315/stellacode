from abc import ABCMeta, abstractmethod


class Surface(metaclass=ABCMeta):
    """A class used to represent an abstract toroidal surface.

    This class can be used to define other child classes.
    If those classes have all of the abstract methods defined here, the Stellacode should work with them.

    This class also implements some concrete methods.
    They allow for the computation of the magnetic field generated by a current carried by the surface.
    They allow for visualization tools.
    """

    @classmethod
    @abstractmethod
    def load_file(cls, pathfile):
        """Instantiate object from a text file.

        :param pathfile: path to file that describes the surface
        :type pathfile: str

        :return: the surface
        :rtype: Surface
        """
        pass


    @abstractmethod
    def get_theta_pertubation(self, compute_curvature):
        """Compute the perturbations of a surface.
        The returned dict must contain the following keys :
        ...

        """
        pass

    @abstractmethod
    def expand_for_plot_part(self):
        """Returns 3 arrays X, Y and Z which can be used to plot the surface.

        :return: tuple of 3 arrays
        :rtype: tuple(array, array, array)
        """
        pass

    def get_B_generated(self, j, positions):
        """Returns the B field generated by a current distribution at given positions.
        positions can have any number of axis, but the xyz coordinates of the points must be in the last one.
        j must have the following shape : (n_pol, n_tor, 3)

        :param j: surface current distribution carried by the surface
        :type j: 3D float array

        :param pos: position where we want to compute B
        :type pos: 1D float array

        :return: B field
        :rtype: 1D float array
        """
        import dask.array as da
        from numpy import expand_dims, float64, linalg, newaxis, pi
        from opt_einsum import contract
        from scipy.constants import mu_0

        from stellacode.tools import eijk, get_rot_tensor

        rot_tensor = get_rot_tensor(self.n_fp)

        def compute_B(grid):
            T = grid[newaxis, newaxis, newaxis, ...] - expand_dims(
                contract("opq,ijq->oijp", rot_tensor, self.P),
                axis=tuple(range(3, 2 + len(grid.shape))),
            )

            K = T / (linalg.norm(T, axis=-1) ** 3)[..., newaxis]

            XYZBs = (
                mu_0 / (4 * pi) * contract("niq,uvq,nuv...j,ijc,uv->...c", rot_tensor, j, K, eijk, self.dS) / self.npts
            )

            return XYZBs

        positions_dask = da.from_array(positions, chunks="auto")

        XYZBs = da.map_blocks(compute_B, positions_dask, dtype=float64).compute()

        return XYZBs

    def get_B_generated_on_surface(self, other, j):
        """Returns the B field generated by a current distribution on another surface.
        j must have the following shape : (n_pol, n_tor, 3)

        :param other: surface on which we want to compute B
        :type other: Surface

        :param j: surface current distribution carried by the surface (on one field period)
        :type j: 3D float array

        :return: B field
        :rtype: 3D float array
        """
        import dask.array as da
        from numpy import float64, linalg, newaxis, pi
        from opt_einsum import contract
        from scipy.constants import mu_0

        from stellacode.tools import eijk, get_rot_tensor

        rot_tensor = get_rot_tensor(self.n_fp)

        def compute_B(XYZgrid):
            T = (
                XYZgrid[newaxis, newaxis, newaxis, ...]
                - contract("opq,ijq->oijp", rot_tensor, self.P)[..., newaxis, newaxis, :]
            )

            K = T / (linalg.norm(T, axis=-1) ** 3)[..., newaxis]

            res = mu_0 / (4 * pi) * contract("niq,uvq,nuvlmj,ijc,uv->lmc", rot_tensor, j, K, eijk, self.dS) / self.npts

            return res

        XYZgrid_dask = da.from_array(other.P, chunks=(10, 10, 3))

        B = da.map_blocks(compute_B, XYZgrid_dask, dtype=float64).compute()

        return B

    def get_B_normal_generated_on_surface(self, other, j):
        """Returns the inward  normal B field generated by a current distribution on another surface.
        j must have the following shape : (n_pol, n_tor, 3)

        :param other: surface on which we want to compute B normal
        :type other: Surface

        :param j: surface current distribution carried by the surface
        :type j: 3D float array

        :return: B normal field
        :rtype: 2D float array
        """
        from numpy import einsum

        return einsum("uvc,cuv->uv", self.get_B_generated_on_surface(other, j), other.n)

    def expand_for_plot_part(self):
        """Returns X, Y, Z arrays of one field period, adding redundancy of first column.

        :return: X, Y, Z arrays
        :rtype: tuple(2D array, 2D array, 2D array)
        """
        from stellacode import np

        shape = self.P.shape[0] + 1, self.P.shape[1]

        X, Y, Z = np.empty(shape), np.empty(shape), np.empty(shape)
        X[:-1:, ::] = self.P[..., 0]
        X[-1, ::] = self.P[0, ::, 0]
        Y[:-1:, ::] = self.P[..., 1]
        Y[-1, ::] = self.P[0, ::, 1]
        Z[:-1:, ::] = self.P[..., 2]
        Z[-1, ::] = self.P[0, ::, 2]

        return X, Y, Z

    def expand_for_plot_whole(self):
        """Returns X, Y, Z arrays of the whole Stellarator.

        :return: X, Y, Z arrays
        :rtype: tuple(2D array, 2D array, 2D array)
        """
        from stellacode import np

        X, Y, Z = self.expand_for_plot_part()
        points = np.stack((X, Y, Z), axis=-1)

        for i in range(1, self.n_fp):
            angle = 2 * i * np.pi / self.n_fp
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            rotated_points = np.einsum("ij,uvj->uvi", rotation_matrix, points)
            X = np.concatenate((X, rotated_points[..., 0]), axis=1)
            Y = np.concatenate((Y, rotated_points[..., 1]), axis=1)
            Z = np.concatenate((Z, rotated_points[..., 2]), axis=1)

        return (
            np.concatenate((X, X[:, 0][:, np.newaxis]), axis=1),
            np.concatenate((Y, Y[:, 0][:, np.newaxis]), axis=1),
            np.concatenate((Z, Z[:, 0][:, np.newaxis]), axis=1),
        )

    def plot_whole_surface(self, representation="surface"):
        """Plots the whole surface.

        :return: None
        :rtype: NoneType
        """
        from mayavi import mlab

        from stellacode import np

        mlab.mesh(*self.expand_for_plot_whole(), representation=representation, colormap="Wistia")
        mlab.plot3d(np.linspace(0, 10, 100), np.zeros(100), np.zeros(100), color=(1, 0, 0))
        mlab.plot3d(np.zeros(100), np.linspace(0, 10, 100), np.zeros(100), color=(0, 1, 0))
        mlab.plot3d(np.zeros(100), np.zeros(100), np.linspace(0, 10, 100), color=(0, 0, 1))
        mlab.show()

    def plot(self, representation="wireframe"):
        """Plots one field period of the surface.

        :return: None
        :rtype: NoneType
        """
        from mayavi import mlab
        from numpy import linspace, zeros

        mlab.mesh(*self.expand_for_plot_part(), representation=representation, colormap="Wistia")
        mlab.plot3d(linspace(0, 6, 100), zeros(100), zeros(100), color=(1, 0, 0))
        mlab.plot3d(zeros(100), linspace(0, 6, 100), zeros(100), color=(0, 1, 0))
        mlab.plot3d(zeros(100), zeros(100), linspace(0, 6, 100), color=(0, 0, 1))
        mlab.show()

    def plot_function_on_surface(self, f):
        """Plots a scalar function on the surface.

        :return: None
        :rtype: NoneType
        """
        from mayavi import mlab
        from numpy import concatenate, linspace, zeros

        fc2 = concatenate((f, f[0:1]), axis=0)
        s = mlab.mesh(*self.expand_for_plot_part(), representation="surface", scalars=fc2)
        mlab.plot3d(linspace(0, 6, 100), zeros(100), zeros(100), color=(1, 0, 0))
        mlab.plot3d(zeros(100), linspace(0, 6, 100), zeros(100), color=(0, 1, 0))
        mlab.plot3d(zeros(100), zeros(100), linspace(0, 6, 100), color=(0, 0, 1))
        mlab.colorbar(s, nb_labels=4, label_fmt="%.1E", orientation="vertical")
        mlab.show()
