from abc import ABC, abstractmethod

class Abstract_shape_gradient(ABC):
    @abstractmethod
    def cost(self,S):
        pass
    @abstractmethod
    def shape_gradient(self,S,theta_pertubation):
        pass
