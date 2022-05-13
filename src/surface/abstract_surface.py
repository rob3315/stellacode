from abc import ABCMeta, abstractmethod


class Surface(metaclass=ABCMeta):
    """
    A class used to represent an abstract toroidal surface.
    """

    @classmethod
    @abstractmethod
    def load_file(cls, pathfile):
        """
        Instantiate object from a text file.
        """
        pass

    @abstractmethod
    def _get_npts(self):
        """
        Get the number of points (in one field period ???).
        """
        pass

    npts = property(_get_npts)

    @abstractmethod
    def _get_nbpts(self):
        """
        Get the tuple of the number of points (in one field period ???).
        """
        pass

    nbpts = property(_get_nbpts)

    @abstractmethod
    def _get_grids(self):
        """
        Get ugrid and vgrid.
        """
        pass

    grids = property(_get_grids)

    @abstractmethod
    def _get_P(self):
        """
        Get all the points of the surface.
        """
        pass

    P = property(_get_P)

    @abstractmethod
    def _get_dpsi(self):
        """
        Get the derivatives of the (u, v) -> (x, y, z) transformation.
        """
        pass

    dpsi = property(_get_dpsi)

    @abstractmethod
    def _get_dS(self):
        """
        Get volume element.
        """
        pass

    dS = property(_get_dS)

    @abstractmethod
    def _get_n(self):
        """
        Get the normal inward unit vectors.
        """
        pass

    n = property(_get_n)

    @abstractmethod
    def _get_principles(self):
        """
        Get the principles.
        """
        pass

    principles = property(_get_principles)

    @abstractmethod
    def _get_I(self):
        """
        Get I.
        """
        pass

    I = property(_get_I)

    @abstractmethod
    def _get_dpsi_uu(self):
        """
        Get dpsi_uu.
        """
        pass

    dpsi_uu = property(_get_dpsi_uu)

    @abstractmethod
    def _get_dpsi_uv(self):
        """
        Get dpsi_uv.
        """
        pass

    dpsi_uv = property(_get_dpsi_uv)

    @abstractmethod
    def _get_dpsi_vv(self):
        """
        Get dpsi_vv.
        """
        pass

    dpsi_vv = property(_get_dpsi_vv)

    @abstractmethod
    def _get_II(self):
        """
        Get II.
        """
        pass

    II = property(_get_II)

    @abstractmethod
    def _get_param(self):
        """
        Get the parametrization of the surface.
        """
        pass

    @abstractmethod
    def _set_param(self):
        """
        Set the parametrization of the surface.
        """

    param = property(_get_param, _set_param)

    @abstractmethod
    def get_theta_pertubation(self, compute_curvature):
        """
        Compute the perturbations of a surface
        """
        pass

    """
    @abstractmethod
    def change_param(param, dcoeff):
        from a surface parameters and an array of modification,
        return the right surface parameters
        :param param: a complex type
        :param dcoeff: the perturbation to apply
        :type dcoeff: 1D array

        where is this thing used?
        
        pass
    """
