from abc import ABC, abstractmethod

class IStrategyDiscriminator(ABC):
    def __init__(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def optimizer(self,*args,**kwargs):
        raise NotImplementedError
    
    
    def discriminate_conjoint_yz(self,*args,**kwargs):
        pass
    
    
    def discriminate_marginal_z(self,*args,**kwargs):
        pass
    
    
    def discriminate_marginal_y(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def architecture(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _get_discriminators(self,*args,**kwargs):
        raise NotImplementedError