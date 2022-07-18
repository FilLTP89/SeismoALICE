import torch
from app.agent.simple.strategy_discriminators import IStrategyDiscriminator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyDiscriminatorWGAN(IStrategyDiscriminator):
    def __init__(self,network,accel,opt,*args, **kwargs):
        
        self.opt    = opt
        self.rlr    = self.opt.config["hparams"]['discriminators.lr']
        self.weight_decay = self.opt.config["hparams"]['discriminators.weight_decay']

        self.Dsyz   = accel(network.DCGAN_DXZ(self.opt.config['Dsyz'],self.opt,model_name='Dsyz')).cuda()
        self.Dsy    = accel(network.DCGAN_Dx(self.opt.config['Dsy'], self.opt,model_name='Dsy')).cuda()
        self.Dszb   = accel(network.DCGAN_Dz(self.opt.config['Dszb'],self.opt,model_name='Dszb')).cuda()
        self._discriminators = [self.Dsyz, self.Dsy, self.Dszb]
        
        super(StrategyDiscriminatorWGAN,self).__init__(*args,**kwargs)

    def _discriminate_conjoint_yz(self,y,y_gen,z, z_gen,*args,**kwargs):
        Dreal_yz = self.Dsyz(y,z_gen)
        Dfake_yz = self.Dsyz(y_gen,z)
        return Dreal_yz, Dfake_yz
    
    def _discriminate_marginal_z(self,z, zr,*args,**kwargs):
        Dreal_z = self.Dszb(z)
        Dfake_z = self.Dszb(zr)
        return Dreal_z, Dfake_z
    
    
    def _discriminate_marginal_y(self,y, yr, *args,**kwargs):
        Dreal_y = self.Dsy(y)
        Dfake_y = self.Dsy(yr)
        return Dreal_y, Dfake_y
    
    def _architecture(self, explore,*args,**kwargs):
        if explore:
            writer_dsyz  = SummaryWriter(self.opt.config['log_dir']['debug.dsyz_writer'])
            writer_dsy   = SummaryWriter(self.opt.config['log_dir']['debug.dsy_writer'])
            writer_dszb  = SummaryWriter(self.opt.config['log_dir']['debug.dszb_writer'])

            writer_dsy.add_graph(next(iter(self.Dsy.children())),torch.randn(10,3,4096).cuda())
            writer_dszb.add_graph(next(iter(self.Dszb.children())),torch.randn(10,1,512).cuda())
            writer_dsyz.add_graph(next(iter(self.Dsyz.children())),(torch.randn(10,3,4096).cuda(),\
                torch.randn(10,1,512).cuda()))
    
    def _get_discriminators(self):
        return self._discriminators
    
    def _optimizer(self):
        return reset_net(self._discriminators,lr = self.rlr, optim='adam', b1=0., b2=0.9, alpha=0.90)