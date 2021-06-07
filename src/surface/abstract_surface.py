from abc import ABC, abstractmethod

class Surface(ABC):
    def __init__(self,surface_parametrization,nbpts,Np):
        """
            A class used to represent an abstract toroidal surface
            ...
            arguments
            ----------
            surface_parametrization : depend on the surface implementation
            nbpts : (int,int)
                nb of toroidal and poloidal point
            Np : intfrom main.surface.surface import Surface
                nb of rotation needed to get the full torus

            Methods
            -------
            load_file(pathfile)
                extract the surface_parametrization from a file
            change_param(param,dcoeff):
                from a surface parameters and an array of modification,
                return the right surface parameters
            plot_surface(self):
                plot the surface
            get_theta_pertubation(self):
                compute the perturbations of a surface
        """
    @abstractmethod
    def load_file(pathfile):
        pass
    @abstractmethod
    def change_param(param,dcoeff):
        pass
    @abstractmethod
    def plot_surface(self):
        pass
    @abstractmethod
    def get_theta_pertubation(self):
        pass