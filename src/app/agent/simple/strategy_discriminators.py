from abc import ABC, abstractmethod

class IStrategyDiscriminator(ABC):
    def __init__(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def _optimizer(self,*args,**kwargs):
        raise NotImplementedError
    
    
    def _discriminate_conjoint_yz(self,*args,**kwargs):
        pass
    
    
    def _discriminate_marginal_z(self,*args,**kwargs):
        pass
    
    
    def _discriminate_marginal_y(self,*args,**kwargs):
        pass
    
    @abstractmethod
    def _architecture(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _get_discriminators(self,*args,**kwargs):
        raise NotImplementedError