import typing as tp

import matplotlib.pyplot as plt
from jax import Array
from jax.typing import ArrayLike

from stellacode import mu_0_fac, np
from stellacode.tools import biot_et_savart, biot_et_savart_op
from stellacode.tools.laplace_force import laplace_force

from .abstract_surface import AbstractBaseFactory, Surface, get_inv_ds_grad
from .current import AbstractCurrent
import jax


class CoilFactory(AbstractBaseFactory):
    """
    Build a coil from a surface and a current class

    Args:
        * current: current class computing a current operator on a given surface grid.
        * build_coils: if True, returns a coilOperator otherwise a CoilSurface
        * compute_grad_current_op: compute also the gradient of the current operator
    """

    current: AbstractCurrent
    build_coils: bool = False
    compute_grad_current_op: bool = False

    @classmethod
    def from_config(cls, config):
        """
        Build a `CoilFactory` instance from a configuration dictionary.

        Args:
            config (dict): The configuration dictionary.

        Returns:
            `CoilFactory`: The built `CoilFactory` instance.
        """
        # Import the necessary functions from the 'imports' module
        from .imports import get_current_potential, get_cws_grid

        # Get the current potential from the configuration
        current = get_current_potential(config)
        # Get the surface from the configuration
        surface = get_cws_grid(config)

        # Construct and return the `CoilFactory` instance
        return cls(surface=surface, current=current)

    def get_trainable_params(self) -> tp.Dict[str, tp.Any]:
        """
        Return the trainable parameters from the current object.

        Returns:
            A dictionary of the trainable parameters, where each key is the name
            of the parameter and the corresponding value is the parameter value.
        """
        return self.current.get_trainable_params()

    def update_params(self, **kwargs):
        """
        Update the parameters of the current class.

        Args:
            **kwargs: Keyword arguments containing the parameters to update.
                Each key should be the name of a trainable parameter and the
                corresponding value is the new value for that parameter.
        """
        # Loop over the keyword arguments
        for k, v in kwargs.items():
            # Check if the key is a trainable parameter of the current class
            if k in type(self.current).model_fields:
                # Update the value of the parameter in the current class
                setattr(self.current, k, v)

    def __call__(self, surface: Surface, **kwargs):
        """
        Construct a `CoilOperator` from a `Surface` and compute the current on the surface.

        Args:
            surface (Surface): The `Surface` object representing the surface on which the current is computed.
            **kwargs: Additional keyword arguments.

        Returns:
            CoilOperator or CoilSurface: The `CoilOperator` object representing the current on the surface,
                or the `CoilSurface` object if `self.build_coils` is `True`.
        """

        # Construct the `CoilOperator` from the surface and compute the current on the surface
        coil_op = CoilOperator.from_surface(
            surface=surface,  # The `Surface` object representing the surface
            current_op=self.current(  # The current operator computed on the surface
                surface.grids, surface.integration_par.max_val_v),
            net_currents=self.current.net_currents,  # The net currents
            # The Fourier coefficients of the surface current
            phi_mn=self.current.get_phi_mn(),
        )

        # Compute the gradient of the current operator if `self.compute_grad_current_op` is `True`
        if self.compute_grad_current_op:
            coil_op.grad_current_op = self.current.get_grad_current_op(
                surface.grids, surface.integration_par.max_val_v)

        # If `self.build_coils` is `True`, return the `CoilSurface` object representing the coil operator,
        # otherwise return the `CoilOperator` object
        if self.build_coils:
            return coil_op.get_coil(phi_mn=self.current.get_phi_mn())
        else:
            return coil_op


class GroovedCoilFactory(AbstractBaseFactory):
    """
    Build a coil from a surface and grooves

    TODO: add boundary element method to this class to compute automatically
    j3d from the result of get_grooves.

    Args:
        * u_ctr_points: poloidal position of the control points of the grooves with dimensions: N_groove + 1 x N_ctr_points
        * v_ctr_points_w: toroidal position weights of the control points of the grooves with dimensions: N_groove x N_ctr_points
        * trainable_params: list of parameters that should be optimized
    """

    u_ctr_points: Array
    v_ctr_points_w: Array
    trainable_params: tp.List[str] = ["v_ctr_points"]

    @classmethod
    def from_params(cls, num_grooves: int, num_ctr_points: int):
        u_pos = np.tile(np.linspace(0, num_ctr_points, endpoint=False)[
                        None], (num_grooves + 1, 1))
        v_pos = np.zeros((num_grooves, num_ctr_points))
        return cls(u_ctr_points=u_pos, v_ctr_points_w=v_pos)

    def get_trainable_params(self):
        return self.trainable_params

    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.trainable_params:
                setattr(self, k, v)

    def __call__(self, surface: Surface, **kwargs):
        j_3d = self.get_j_3d(surface.grids)
        j_surface = np.einsum("klab,kla,kl->klb",
                              surface.get_g_upper_basis(), j_3d, surface.ds)
        coil = CoilSurface.from_surface(surface=surface)
        coil.j_surface = j_surface
        coil.j_3d = j_3d

        return coil

    def get_j_3d(self, grids):
        """Compute j_3d on surface grid points from the groove parameters using the Boundary Element Method"""
        raise NotImplementedError

    def get_grooves(self, u):
        """
        Return a list of values for v corresponding to the position of grooves on the surface.
        There may be advantages (and disadvantages) in using cubic interpolation, which is implemented for jax here:
        https://jax-cosmo.readthedocs.io/en/latest/_modules/jax_cosmo/scipy/interpolate.html

        Note that the first 'groove' is defined by v=0 and the last groove is defined by v=1.
        """
        v_ctr_points = np.cumsum(jax.nn.softmax(self.v_ctr_points_w), axis=0)
        v_ctr_points = np.concatenate(
            (np.zeros_like(v_ctr_points[:1]), v_ctr_points), axis=0)
        return jax.vmap(np.interp, in_axes=(None, 0, 0))(u, self.u_ctr_points, v_ctr_points)


class CoilOperator(Surface):
    """Represent a coil operator

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
    """

    current_op: ArrayLike
    grad_current_op: tp.Optional[
        ArrayLike
    ] = None  # Dimensions of returned op are: dimensions: Ncurrent_op x Nu x Nv x N_j_surf x N_grad
    net_currents: tp.Optional[ArrayLike] = None
    phi_mn: tp.Optional[ArrayLike] = None

    @classmethod
    def from_surface(
        cls,
        surface: Surface,
        current_op: ArrayLike,
        net_currents: tp.Optional[ArrayLike] = None,
        phi_mn: tp.Optional[ArrayLike] = None,
    ) -> "CoilOperator":
        """
        Create a CoilOperator object from a Surface object.

        Args:
            surface (Surface): Surface object on which to create a CoilOperator.
            current_op (ArrayLike): Tensor for computing the 2D surface current from the current weights.
            net_currents (ArrayLike, optional): Net poloidal and toroidal currents. Defaults to None.
            phi_mn (ArrayLike, optional): Vector of Fourier coefficients of the surface current.
                If None, use self.phi_mn. Defaults to None.

        Returns:
            CoilOperator: CoilOperator object created from the input parameters.
        """
        # Create a dictionary of the surface attributes that are also in CoilOperator's model_fields
        dict_ = {k: v for k, v in dict(
            surface).items() if k in cls.model_fields.keys()}
        # Add the current_op attribute to the dictionary
        dict_["current_op"] = current_op
        # Add the net_currents attribute to the dictionary
        dict_["net_currents"] = net_currents
        # Add the phi_mn attribute to the dictionary
        dict_["phi_mn"] = phi_mn
        # Create and return a CoilOperator object from the dictionary
        return cls(**dict_)

    def get_coil(self, phi_mn: tp.Optional[ArrayLike] = None):
        """
        Return a CoilSurface object representing the coil operator.

        Parameters
        ----------
        phi_mn : ArrayLike, optional
            Vector of Fourier coefficients of the surface current.
            If None, use self.phi_mn.

        Returns
        -------
        CoilSurface
            CoilSurface object representing the coil operator.
        """
        # Create a dictionary of all attributes except current_op and grad_current_op
        dict_ = {k: v for k, v in dict(self).items() if k not in [
            "current_op", "grad_current_op"]}

        # If jac_xyz is not None, compute the surface and 3D current
        if self.jac_xyz is not None:
            dict_["j_surface"] = self.get_j_surface(
                phi_mn)  # Compute the surface current
            dict_["j_3d"] = self.get_j_3d(phi_mn)  # Compute the 3D current

            # If grad_current_op is not None, compute the gradient of the surface and 3D currents
            if self.grad_current_op is not None:
                dict_["grad_j_surface"] = self.get_grad_j_surface(phi_mn)
                dict_["grad_j_3d"] = self.get_grad_j_3d(phi_mn)

        # Return a CoilSurface object with the computed attributes
        return CoilSurface(**dict_)

    def get_j_surface(self, phi_mn: tp.Optional[ArrayLike] = None):
        """
        Compute the contravariant components of the surface current.

        Parameters
        ----------
        phi_mn : ArrayLike, optional
            Vector of Fourier coefficients of the surface current.
            If None, use self.phi_mn.

        Returns
        -------
        numpy.ndarray
            Contravariant components of the surface current.
            Dimensions: Nu x Nv x N_current_op
        """
        # If phi_mn is not provided, use self.phi_mn
        if phi_mn is None:
            phi_mn = self.phi_mn

        # Compute the contravariant components of the surface current
        #   1) Multiply by the current operator
        #   2) Multiply by the vector of Fourier coefficients
        return np.einsum("oijk,o->ijk", self.current_op, phi_mn)

    def get_grad_j_surface(self, phi_mn: tp.Optional[ArrayLike] = None):
        """
        Compute the gradient of the contravariant components of the surface current.

        Parameters
        ----------
        phi_mn : ArrayLike, optional
            Vector of Fourier coefficients of the surface current.
            If None, use self.phi_mn.

        Returns
        -------
        numpy.ndarray
            Gradient of the surface current.
            Dimensions: Nu x Nv x N_current_op x N_j_surf
        """
        if phi_mn is None:
            phi_mn = self.phi_mn
        # Compute the gradient of the surface current
        #   1) Multiply by the gradient operator
        #   2) Multiply by the vector of Fourier coefficients
        return np.einsum("oijkl,o->ijkl", self.grad_current_op, phi_mn)

    def get_j_3d(self, phi_mn: tp.Optional[ArrayLike] = None, scale_by_ds: bool = True):
        """
        Compute the 3D current onto the surface

        Args:
            * phi_mn: vector of fourier coefficients of the surface current (optional, default: self.phi_mn)
            * scale_by_ds: whether to scale the result by 1/ds (optional, default: True)

        Returns:
            * 3D current onto the surface (Nu x Nv x 3)
        """
        assert (self.ds is not None)

        # If phi_mn is not provided, use self.phi_mn
        if phi_mn is None:
            phi_mn = self.phi_mn

        # If scale_by_ds is True, scale the result by 1/ds
        if scale_by_ds:
            return np.einsum("oijk,ijdk,ij,o->ijd", self.current_op, self.jac_xyz, 1 / self.ds, phi_mn)
        else:
            return np.einsum("oijk,ijdk,o->ijd", self.current_op, self.jac_xyz, phi_mn)

    def get_grad_j_3d(self, phi_mn: tp.Optional[ArrayLike] = None):
        """
        Compute the gradient of the 3D current versus u and v

        Args:
            * phi_mn: vector of fourier coefficients of the surface current (optional, default: self.phi_mn)

        Returns:
            * gradient of the 3D current (Nu x Nv x 3 x 2)
        """
        assert (self.ds is not None)

        if phi_mn is None:
            phi_mn = self.phi_mn

        # Compute the gradient of the surface current
        grad_j_surface = self.get_grad_j_surface(phi_mn)

        # Compute the gradient of 1/ds
        invds_grad = get_inv_ds_grad(self)

        # Compute the gradient of the 3D current
        #   1) Compute the gradient of the surface current wrt u and v
        #   2) Multiply by the Jacobian matrix
        #   3) Multiply by 1/ds
        #   4) Add the contribution of the Hessian of the surface current
        #   5) Add the contribution of the gradient of 1/ds
        fac1 = np.einsum("ijkl,ijdk,ij->ijdl", grad_j_surface,
                         self.jac_xyz, 1 / self.ds)
        fac2 = np.einsum("oijk,ijdkl,ij,o->ijdl", self.current_op,
                         self.hess_xyz, 1 / self.ds, phi_mn)
        fac3 = np.einsum("oijk,ijdk,ijl,o->ijdl", self.current_op,
                         self.jac_xyz, invds_grad, phi_mn)
        return fac1 + fac2 + fac3

    def get_b_field_op(
        self,
        xyz_plasma: ArrayLike,
        plasma_normal: tp.Optional[ArrayLike] = None,
        use_mu_0_factor: bool = True
    ) -> ArrayLike:
        """
        Compute the magnetic field operator using the Biot et Savart operator

        Args:
            xyz_plasma (ArrayLike): Surface on which the magnetic field is computed
            plasma_normal (Optional[ArrayLike], optional): The plasma normal vector,
                against which is computed the normal magnetic field. Defaults to None.
            use_mu_0_factor (bool, optional): Whether to multiply the result by mu_0/4pi.
                Defaults to True.

        Returns:
            ArrayLike: The magnetic field operator
        """
        # Compute the magnetic field operator using the Biot-Savart operator
        bs_op = biot_et_savart_op(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            surface_current=self.current_op,
            jac_xyz_coil=self.jac_xyz,
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )

        # Multiply the magnetic field operator by mu_0/4pi if use_mu_0_factor is True
        if use_mu_0_factor:
            return bs_op * mu_0_fac
        else:
            return bs_op

    def get_current_basis_dot_prod(self):
        """Compute the scalar product matrix of the current basis functions

        This function computes the scalar product matrix of the current basis
        functions. The scalar product matrix is a tensor that specifies the
        scalar product of the current basis functions with themselves.

        Returns:
            numpy.ndarray: The scalar product matrix of the current basis
                functions.
        """
        # Compute the scalar product matrix using einsum
        return (
            np.einsum(
                "oija,ijda,ijdk,pijk,ij->op",  # einsum formula
                self.current_op,  # current op tensor
                self.jac_xyz,  # jacobian tensor
                self.jac_xyz,  # jacobian tensor
                self.current_op,  # current op tensor
                1 / self.ds,  # inverse of ds
                optimize=True,  # enable optimization
            )
            * self.dudv  # multiply by dudv
        )

    def plot_j_surface(self, num_prec: int = 2, ax=None):
        return self.plot_2d_field(self.get_j_3d()[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)


class CoilSurface(Surface):
    """Represent a coil

    Args:
        * current_op: tensor for computing the 2D surface current from the current weights
        * j_surface: contravariant components of the current: J^i
        * net_currents: net poloidal and toroidal currents
        * nfp: number of toroidal periods (Note: should be in a specialized class instead of a general coil class)
        * phi_mn: current weights
    """

    j_surface: tp.Optional[Array] = None
    j_3d: tp.Optional[Array] = None
    net_currents: tp.Optional[Array] = None
    grad_j_surface: tp.Optional[Array] = None
    grad_j_3d: tp.Optional[Array] = None

    @classmethod
    def from_surface(cls, surface: Surface):
        return cls(**{k: v for k, v in dict(surface).items() if k in cls.model_fields.keys()})

    @property
    def field_keys(self):
        """
        Compute the keys of the fields associated with the coil surface.

        Returns:
            list[str]: List of field keys.
        """
        # Concatenate the superclass field keys with the additional keys
        return super().field_keys + [
            "j_surface",  # contravariant components of the current: J^i
            "j_3d"  # 3D current
        ]

    def get_b_field(
        self,
        xyz_plasma: ArrayLike,
        plasma_normal: tp.Optional[ArrayLike] = None,
        use_mu_0_factor: bool = True
    ) -> ArrayLike:
        """
        Compute the magnetic field.

        Args:
            xyz_plasma (ArrayLike): Surface on which the magnetic field is computed.
            plasma_normal (Optional[ArrayLike], optional): The plasma normal vector,
                against which is computed the normal magnetic field. Defaults to None.
            use_mu_0_factor (bool, optional): Whether to multiply the result by mu_0/4pi.
                Defaults to True.

        Returns:
            ArrayLike: The magnetic field.
        """
        assert (self.ds is not None)
        # Compute the magnetic field using the Biot-Savart law.
        # The magnetic field is computed on the surface of the plasma.
        bf = biot_et_savart(
            xyz_plasma=xyz_plasma,
            xyz_coil=self.xyz,
            # Current density multiplied by the element size.
            j_3d=self.j_3d * self.ds[..., None],
            dudv=self.dudv,
            plasma_normal=plasma_normal,
        )

        # Multiply the magnetic field by mu_0/4pi if use_mu_0_factor is True.
        # This is done to convert the magnetic field from units of Tesla to Webers.
        if use_mu_0_factor:
            return bf * mu_0_fac
        else:
            return bf

    def get_j_surface(self):
        """
        Get the contravariant components of the surface current.

        Returns:
            ArrayLike: The contravariant components of the surface current.
        """
        # Compute and return the contravariant components of the surface current.
        # The contravariant components are the components of the surface current vector
        # in the basis defined by the surface tangent vectors.
        return self.j_surface

    def naive_laplace_force(self, epsilon: float = 1.0):
        """
        Naive computation of the Laplace force.

        This method computes the Laplace force of a distribution of currents on itself.
        It does so by computing the average magnetic field at two points displaced by plus or minus
        `epsilon` times the minimal distance between two neighboring points in the direction of the normal unit vector.
        The magnetic field is computed at these two points using the `get_b_field` method.
        The Laplace force is then computed as the cross product of the surface current and
        the average magnetic field.

        Args:
            epsilon (float, optional): Factor of the distance at which the magnetic field is computed
                in unit of inter points. Should be greater than 1 otherwise the computation may be very inaccurate. Defaults to 1.0.

        Returns:
            numpy.ndarray: The naive Laplace force of the surface current.
        """
        # Get the surface current
        j_3d = self.j_3d
        # Compute the minimum distance between two neighboring points
        dist = np.min(np.linalg.norm(self.xyz[1:] - self.xyz[:-1], axis=-1))
        # Compute the displaced points
        xyz_ext = self.xyz + epsilon * dist * self.normal_unit
        xyz_int = self.xyz - epsilon * dist * self.normal_unit
        # Compute the average magnetic field at the displaced points
        b_avg = self.get_b_field(xyz_ext) + self.get_b_field(xyz_int)
        # Compute the naive Laplace force
        return 0.5 * np.cross(j_3d, b_avg)

    def laplace_force(
        self,
        cut_coils: tp.Optional[tp.List[int]] = None,
        num_tor_pts: int = 100000,
        end_u: int = 1000000,
        end_v: int = 1000000,
    ):
        """
        Compute the Laplace force of a distribution of currents on itself.

        Args:
            * cut_coils: list of indexes of coils to cut. If None, compute the force on the full surface.
            * num_tor_pts: number of toroidal points to use for the force computation.
            * end_u: cut the points along u at end_u (this parameter is there for checking the coherence with older implementations).
            * end_v: cut the points along v at end_v.

        Returns:
            * array of shape (3,) containing the force along the x, y, z axes.
        """
        # Get the upper basis function
        g_up = self.get_g_upper_basis()

        assert (self.j_3d is not None)

        # If no cuts are specified, compute the force on the full surface
        if cut_coils is None:
            return laplace_force(
                j_3d_f=self.j_3d[:, :num_tor_pts],
                xyz_f=self.xyz[:, :num_tor_pts],
                j_3d_b=self.j_3d,
                xyz_b=self.xyz,
                normal_unit_b=self.normal_unit,
                ds_b=self.ds,
                g_up_map_b=g_up,
                du=self.du,
                dv=self.dv,
                end_u=end_u,
                end_v=end_v,
            )
        else:
            assert (self.normal_unit is not None)
            assert (self.ds is not None)

            # Compute the Laplace force for each coil segment
            lap_forces = []
            begin = 0
            _cut_coils = cut_coils + [1000000]
            for end in _cut_coils:
                lap_forces.append(
                    laplace_force(
                        j_3d_f=self.j_3d[:, :num_tor_pts],
                        xyz_f=self.xyz[:, :num_tor_pts],
                        j_3d_b=self.j_3d[:, begin:end],
                        xyz_b=self.xyz[:, begin:end],
                        normal_unit_b=self.normal_unit[:, begin:end],
                        ds_b=self.ds[:, begin:end],
                        g_up_map_b=g_up[:, begin:end],
                        du=self.du,
                        dv=self.dv,
                    )
                )
                begin = end
            # Sum the forces from all segments
            return sum(lap_forces)

    def imshow_j(self):
        """
        Show the current surface density using imshow.

        This function plots the magnitude of the current density on the
        coil winding surface using the imshow function from matplotlib.

        Parameters:
            None

        Returns:
            None
        """
        # Compute the magnitude of the current density in the last axis
        j_mag = np.linalg.norm(self.j_3d, axis=-1)

        # Plot the magnitude of the current density using imshow
        plt.imshow(j_mag, cmap="seismic")

    def plot_j_surface(self, num_prec: int = 2, ax=None):
        """
        Plots the current density on the coil winding surface.

        Parameters:
            - num_prec: an integer representing the precision of the plot (default is 2).
            - ax: a matplotlib axis object to plot the field on (default is None).

        Returns:
            The matplotlib axis object with the current density plot.
        """
        assert (self.j_3d is not None)
        return self.plot_2d_field(self.j_3d[:, : self.nbpts[1]], num_prec=num_prec, ax=ax)
