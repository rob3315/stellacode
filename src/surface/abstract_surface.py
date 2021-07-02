from abc import ABC, abstractmethod

class Surface(ABC):
    """A class used to represent an abstract toroidal surface
    """

    def __init__(self,surface_parametrization,nbpts,Np):
        """
        :param surface_parametrization:  depend on the surface implementation
        :param nbpts: nb of toroidal and poloidal points
        :type nbpts: (int,int)
        :param Np: nb of rotation needed to get the full torus
        :type Np: int
        """
        pass
    @abstractmethod
    def load_file(pathfile):
        """extract the surface_parametrization from a file
        """
        pass
    @abstractmethod
    def change_param(param,dcoeff):
        """from a surface parameters and an array of modification,
        return the right surface parameters
        :param param: a complex type
        :param dcoeff: the perturbation to apply
        :type dcoeff: 1D array
        """
        pass
    @abstractmethod
    def get_theta_pertubation(self):
        """compute the perturbations of a surface
        """
        pass