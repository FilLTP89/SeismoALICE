import torch
from torch.utils.tensorboard import SummaryWriter
from core.trainer.basic_frame import Agent
from common.common_nn import reset_net, set_weights
from configuration import app

class Generators(Agent):
    def __init__(self,network,config,logger, accel,opt, gradients_tracker,debug_writer,
        strategy, *args, **kwargs):

        self.current_val = 0
        self.config = config
        self.opt    = opt
        self.generators = []
        self.gradients_tracker = gradients_tracker
        self.debug_writer = debug_writer

        self.strategy = strategy(network,accel,opt)
        self.generators = self.strategy._get_generators()
        self.optimizer_encoder = self.strategy._optimizer_encoder()
        self.optimizer_decoder = self.strategy._optimizer_decoder()

        self._architecture(app.EXPLORE)
        super(Generators,self).__init__(self.generators, [self.optimizer_encoder,
                self.optimizer_decoder], config, logger, accel,*args, **kwargs)
    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.generators,epoch)
    
    def track_weight(self, epoch):
        for gen in self.generators:
            try:
                self.track_weight_change(writer= self.debug_writer, tag = 'gen', model= gen.module.eval(),epoch = epoch)
            except Exception:
                try:
                    self.track_weight_change(writer= self.debug_writer, tag= 'F[cnn_common]', model= gen.module.cnn_common.eval(),epoch = epoch)
                    self.track_weight_change(writer= self.debug_writer, tag= 'F[cnn_broadband]', model= gen.module.cnn_broadband.eval(),epoch = epoch)
                    self.track_weight_change(writer= self.debug_writer, tag= 'F[master]', model= gen.module.master.eval(),epoch = epoch)
                except Exception:
                    raise AttributeError(f"The generator doesn't have this specified attribute")
    
    def _architecture(self,explore):
        self.strategy._architecture(explore)
    
    def __getattr__(self,name):
        if name in self.strategy._get_name_generators():
            disc = getattr(self.strategy,name, None)
            return disc
        else:
            raise ValueError(f"The generators agent doesn't have the attribute {name}")
