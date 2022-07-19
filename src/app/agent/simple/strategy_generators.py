from abc import ABC, abstractmethod

class IStrategyGenerator(ABC):
    def __init__(self,*args,**kwargs):
        pass

    @abstractmethod
    def _optimizer_encoder(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _optimizer_decoder(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _architecture(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _get_generators(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_name_generators(self):
        raise NotImplementedError