import torch
from app.agent.unic.strategy_discriminators import IStrategyDiscriminator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyDiscriminatorBiCycleGAN(IStrategyDiscriminator):
    def __init__(self,network, accel, opt, *args, **kwargs):
        self.opt = opt
        self.rlr    = self.opt.config["hparams"]['discriminators.lr']
        self.weight_decay = self.opt.config["hparams"]['discriminators.weight_decay']

        self.DVAE   = accel(network.DCGAN_Dx(self.opt.config['DVAE'], self.opt,model_name='DVEA')).cuda()
        self.DLR    = accel(network.DCGAN_Dx(self.opt.config['DLR'],  self.opt,model_name='DLR')).cuda()

        self._discriminators        = [self.DVAE, self.DLR]
        self._name_discriminators   = ['DVEA', 'DLR']
        self.optimizer_vae, self.optimizer_lr = self._optimizer()
        
        super(StrategyDiscriminatorBiCycleGAN, self).__init__(*args, **kwargs)
    
    def _architecture(self, explore,*args,**kwargs):
        if explore:
            writer_dvae  = SummaryWriter(self.opt.config['log_dir']['debug.dvae_writer'])
            writer_dlr   = SummaryWriter(self.opt.config['log_dir']['debug.dlr_writer'])
            writer_dvae.add_graph(next(iter(self.DVAE.children())),(torch.randn(10,3,4096).cuda(), torch.randn(10,3,4096).cuda()))
            writer_dlr.add_graph(next(iter(self.DVAE.children())),(torch.randn(10,3,4096).cuda(), torch.randn(10,3,4096).cuda()))

    def _get_discriminators(self):
        return self._discriminators

    def _get_name_discriminators(self):
        return self._name_discriminators
    
    def _discriminate_marginal_y(self, y, yr):
        return self.DVAE(y), self.DVAE(yr)
    
    def _discriminate_marginal_zd(self,z, zr):
        return self.DLR(z), self.DLR(zr)
    
    def _optimizer(self):
        optimizer_vae   = reset_net([self.DVAE],lr = self.rlr, optim='adam', b1=0.5, b2=0.999, alpha=0.90)
        optimizer_lr    = reset_net([self.DLR], lr = self.rlr, optim='adam', b1=0.5, b2=0.999, alpha=0.90)
        return optimizer_vae, optimizer_lr