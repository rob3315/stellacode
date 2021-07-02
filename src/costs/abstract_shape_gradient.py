from abc import ABC, abstractmethod

class Abstract_shape_gradient(ABC):
    """An Abstract_shape_gradient is the interface for any cost
        
    :param S: a surface
    :type S: for now only Surface_Fourier are supported
    """
    @abstractmethod
    def cost(self,S):

        pass
    @abstractmethod
    def shape_gradient(self,S,theta_pertubation):
        """The implementation of the shape gradient

        :param S: a surface
        :type S: for now only Surface_Fourier are supported
        :param theta_pertubation: see `get_theta_perturbation of a surface`
        :type theta_pertubation: dictionary
        """
        pass
