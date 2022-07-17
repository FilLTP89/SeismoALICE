import torch
from core.trainer.basic_frame import Agent
from torch.utils.tensorboard import SummaryWriter
from common.common_nn import reset_net,zcat
from tools.generate_noise import noise_generator
from configuration import app
class Discriminators(Agent):
    def __init__(self,network,config,logger, accel, opt, gradients_tracker,debug_writer,
                strategy, *args, **kwargs):
        self.config = config
        self.opt    = opt
        self.std    = 1.0
        self.debug_writer       = debug_writer
        self.discriminators     = []
        self.gradients_tracker  = gradients_tracker

        self.strategy   = strategy(network,accel,opt)
        self.optimizer  = self.strategy._optimizer()
        self.discriminators = self.strategy._get_discriminators()

        self.architecture(app.EXPLORE)
        super(Discriminators,self).__init__(self.discriminators, self.optimizer, config, logger, accel,*args, **kwargs)
    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.discriminators,epoch)
    
    def track_weight(self,epoch):
        for net in self.discriminators:
            self.track_weight_change(writer =  self.debug_writer, tag = net.module.model_name,
            model= net.module.cnn.eval(), epoch = epoch)
    
    def architecture(self, explore):
        self.strategy._architecture(explore)
    
    def discriminate_conjoint_yz(self,y,yr,z,zr):
        Dreal, Dfake = self.strategy._discriminate_conjoint_yz(y,yr,z,zr)
        return Dreal,Dfake 

    def discriminate_marginal_y(self,y,yr):
        Dreal, Dfake = self.strategy._discriminate_marginal_y(y, yr)
        return Dreal, Dfake
    
    def discriminate_marginal_zd(self,z,zr):
        Dreal, Dfake = self.strategy._discriminate_marginal_z(z,zr)
        return Dreal, Dfake
    
    def discriminate_crosss_entropy_y(self,y, yr):
        Dreal, Dfake = self.strategy._discriminate_crosss_entropy_y(y,yr)
        return Dreal, Dfake
    
    def discriminate_crosss_entropy_zd(self,z,zr):
        Dreal, Dfake = self.strategy._discriminate_cross_entropy_zd(z,zr)
        return Dreal, Dfake