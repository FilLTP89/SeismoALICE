from abc import ABC, abstractmethod

class IStrategyDiscriminator(ABC):
    def __init__(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def discriminate_conjoint_yz(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def discriminate_marginal_z(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def discriminate_marginal_y(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def architecture(self,*args,**kwargs):
        raise NotImplementedError