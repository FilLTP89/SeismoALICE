import torch
from app.agent.simple.strategy_discriminators import IStrategyDiscriminator
from common.common_nn import zcat
from torch.utils.tensorboard import SummaryWriter
class StrategyDiscriminatorWGAN(IStrategyDiscriminator):
    def __init__(self,network,config,accel,opt,*args, **kwargs):
        self.opt    = opt
        self.Dsyz   = accel(network.DCGAN_Dx(self.opt.config['Dsyz'],self.opt,model_name='Dsy')).cuda()
        self.Dsy    = accel(network.DCGAN_Dx(self.opt.config['Dsy'], self.opt,model_name='Dsy')).cuda()
        self.Dszb   = accel(network.DCGAN_Dz(self.opt.config['Dszb'],self.opt,model_name='Dszb')).cuda()
        super(StrategyDiscriminatorWGAN,self).__init__(*args,**kwargs)

    def discriminate_conjoint_yz(self,y,z, y_gen,*args,**kwargs):
        Dreal_yz = self.Dsyz(zcat(self.Dsy(y), self.Dszb(z)))
        Dfake_yz = self.Dsyz(zcat(self.Dsy(y_gen), self.Dszb(z)))
        return Dreal_yz, Dfake_yz
    
    
    def discriminate_marginal_z(self,z, zr,*args,**kwargs):
        Dreal_z = self.Dszb(z)
        Dfake_z = self.Dszb(zr)
        return Dreal_z, Dfake_z
    
    
    def discriminate_marginal_y(self,y, yr, *args,**kwargs):
        Dreal_y = self.Dsy(y)
        Dfake_y = self.Dsy(yr)
        return Dreal_y, Dfake_y
    
    
    def architecture(self, explore,*args,**kwargs):
        if explore: 
            writer_dsy   = SummaryWriter(self.opt.config['log_dir']['debug.dsy_writer'])
            writer_dszb  = SummaryWriter(self.opt.config['log_dir']['debug.dszb_writer'])

            writer_dsy.add_graph(next(iter(self.Dsy.children())),torch.randn(10,3,4096).cuda())
            writer_dszb.add_graph(next(iter(self.Dszb.children())),torch.randn(10,1,512).cuda())