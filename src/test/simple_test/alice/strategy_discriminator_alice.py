import torch
from app.agent.simple.strategy_discriminators import IStrategyDiscriminator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyDiscriminatorALICE(IStrategyDiscriminator):
    def __init__(self,network,accel,opt,*args, **kwargs):
        self.opt    = opt
        self.rlr    = self.opt.config["hparams"]['discriminators.lr']
        self.weight_decay = self.opt.config["hparams"]['discriminators.weight_decay']
        self.Dyz    = accel(network.DCGAN_DXZ(self.opt.config['Dyz'],self.opt,model_name='Dyz')).cuda()
        self.Dsy    = accel(network.DCGAN_Dx(self.opt.config['Dsy'], self.opt,model_name='Dsy')).cuda()
        self.Dszb   = accel(network.DCGAN_Dz(self.opt.config['Dszb'],self.opt,model_name='Dszb')).cuda()
        self.Dzzb   = accel(network.DCGAN_Dz(self.opt.config['Dzzb'],self.opt,model_name='Dzzb')).cuda()
        self.Dyy    = accel(network.DCGAN_Dx(self.opt.config['Dyy'],self.opt, model_name='Dyy')).cuda()

        self._discriminators = [self.Dyz,self.Dzzb,self.Dyy, self.Dsy, self.Dszb]
        self._name_discriminators = ['Dyz','Dzzb','Dyy','Dsy', 'Dszb']
        super(StrategyDiscriminatorALICE,self).__init__(*args,**kwargs)

    def _discriminate_conjoint_yz(self,y,y_gen,z, z_gen,*args,**kwargs):
        Dreal_yz = self.Dyz(zcat(self.Dsy(y), self.Dszb(z_gen)))
        Dfake_yz = self.Dyz(zcat(self.Dsy(y_gen), self.Dszb(z)))
        return Dreal_yz, Dfake_yz
    
    def discriminate_crosss_entropy_z(self,z, zr,*args,**kwargs):
        Dreal_z = self.Dzzb(zcat(z,z))
        Dfake_z = self.Dzzb(zcat(z,zr))
        return Dreal_z, Dfake_z
    
    
    def discriminate_crosss_entropy_y(self,y, yr, *args,**kwargs):
        Dreal_y = self.Dyy(y,y)
        Dfake_y = self.Dyy(y,yr)
        return Dreal_y, Dfake_y
    
    def _architecture(self, explore,*args,**kwargs):
        if explore: 
            writer_dyz  = SummaryWriter(self.opt.config['log_dir']['debug.dyz_writer'])
            writer_dzz  = SummaryWriter(self.opt.config['log_dir']['debug.dzz_writer'])
            writer_dyy  = SummaryWriter(self.opt.config['log_dir']['debug.dyy_writer'])

            writer_dsy   = SummaryWriter(self.opt.config['log_dir']['debug.dsy_writer'])
            writer_dszb  = SummaryWriter(self.opt.config['log_dir']['debug.dszb_writer'])

            writer_dyz.add_graph(next(iter(self.Dyz.children())), torch.randn(10,128,64).cuda())
            writer_dzz.add_graph(next(iter(self.Dzzb.children())),torch.randn(10,2,128).cuda())
            writer_dyy.add_graph(next(iter(self.Dyy.children())), torch.randn(10,2,4096).cuda())
            writer_dsy.add_graph(next(iter(self.Dsy.children())), torch.randn(10,6,4096).cuda())
            writer_dszb.add_graph(next(iter(self.Dszb.children())),torch.randn(10,2,512).cuda())
    
    def _get_discriminators(self):
        return self._discriminators
    
    def _get_name_discriminators(self):
        return self._name_discriminators
    
    def _optimizer(self):
        return reset_net(self._discriminators,lr = self.rlr, optim='adam', b1=0., b2=0.9,
            alpha=0.90, weights_decay=self.weight_decay)