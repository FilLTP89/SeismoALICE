from core.trainer.basic_frame import Agent
from configuration import app

class Discriminators(Agent):
    def __init__(self,network,config,logger, accel, opt, gradients_tracker,
                debug_writer,strategy,*args, **kwargs):
        self.config = config
        self.opt    = opt
        self.std    = 1.0
        self.debug_writer = debug_writer
        self.gradients_tracker = gradients_tracker
        self.discriminators  = []

        self.strategy   = strategy(network,accel,opt)
        self.optimizer  = self.strategy._optimizer()
        self.discriminators = self.strategy._get_discriminators()

        super(Discriminators,self).__init__(self.discriminators, self.optimizer, config, logger,
            accel,*args, **kwargs)
    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.discriminators,epoch)
    
    def track_weight(self,epoch):
        for net in self.discriminators:
            self.track_weight_change(writer =  self.debug_writer, tag = net.module.model_name,
            model= net.module,epoch = epoch)
    
    def discriminate_conjointe_xy(self,x,y,yr):
        return self.strategy._discriminate_conjointe_xy(x,y,yr)
        
    def discriminate_xz(self,x,xr,z,zr):
        return self.strategy._discriminate_xz(x,xr,z,zr)
    
    def discriminate_yz(self,y,yr,z,zr):
        return self.strategy._discriminate_yz(y,yr,z,zr)

    def discriminate_marginal_y(self,y,yr):
        return self.strategy._discriminate_marginal_y(y,yr)
    
    def discriminate_marginal_zd(self,z,zr):
        return self.strategy._discriminate_marginal_zd(z,zr)
    
    def discriminate_cross_entropy_y(self,y,yr):
        return self.strategy._discriminate_cross_entropy_y(y,yr)
    
    def discriminate_cross_entropy_x(self,x,xr):
        return self.strategy.discriminate_cross_entropy_x(x,xr) 
    
    def discriminate_cross_entropy_zd(self,z,zr):
        return self.strategy._discriminate_cross_entropy_zd(z,zr)
    
    def discriminate_cross_entropy_zf(self,z,zr):
        return self.strategy.discriminate_cross_entropy_zf(z,zr)

    def discriminate_marginal_x(self,x,xr):
        return self.strategy._discriminate_marginal_x(x,xr)
    
    def discriminate_marginal_zxy(self,zxy,zxyr):
        return self.strategy._discriminate_marginal_zxy(zxy,zxyr)
    
    def __getattr__(self,name):
        if name in self.strategy._get_name_discriminators():
            disc = getattr(self.strategy,name, None)
            return disc
        else:
            raise ValueError(f"The discriminator agent doesn't have the attribute {name}")

    
    
    
