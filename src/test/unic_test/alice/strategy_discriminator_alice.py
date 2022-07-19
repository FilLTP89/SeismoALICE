import torch
from app.agent.unic.strategy_discriminators import IStrategyDiscriminator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyDiscriminatorALICE(IStrategyDiscriminator):
    def __init__(self,network,accel,opt,*args, **kwargs):
        self.rlr = self.opt.config["hparams"]['discriminators.lr']
        self.weight_decay = self.opt.config["hparams"]['discriminators.weight_decay']

        self.Dy     = accel(network.DCGAN_Dx( self.opt.config['Dy'],  self.opt,model_name='Dy')).cuda()
        self.Dsy    = accel(network.DCGAN_DXZ(self.opt.config['Dsy'], self.opt,model_name='Dsy')).cuda()
        self.Dzb    = accel(network.DCGAN_Dz( self.opt.config['Dzb'], self.opt,model_name='Dzb')).cuda()
        self.Dszb   = accel(network.DCGAN_DXZ(self.opt.config['Dszb'],self.opt,model_name='Dszb')).cuda()
        self.Dyz    = accel(network.DCGAN_DXZ(self.opt.config['Dyz'], self.opt,model_name='Dyz')).cuda()

        self.Dx     = accel(network.DCGAN_Dx( self.opt.config['Dx'],  self.opt,model_name='Dx')).cuda()
        self.Dsx    = accel(network.DCGAN_DXZ(self.opt.config['Dsx'], self.opt,model_name='Dsx')).cuda()
        self.Dzf    = accel(network.DCGAN_Dz( self.opt.config['Dzf'], self.opt,model_name='Dzf')).cuda()
        self.Dszf   = accel(network.DCGAN_DXZ(self.opt.config['Dszf'],self.opt,model_name='Dszf')).cuda()
        self.Dxz    = accel(network.DCGAN_DXZ(self.opt.config['Dxz'], self.opt,model_name='Dxz')).cuda()

        self.discriminators  = [ self.Dy, self.Dsy, self.Dzb, self.Dszb, self.Dyz, 
                        self.Dx, self.Dsx, self.Dzf, self.Dszf, self.Dxz]
        
        self._name_discriminators = ['Dy','Dsy', 'Dzb', 'Dszb', 'Dyz', 'Dx', 'Dsx', 'Dzf', 'Dszf', 'Dxz']
        super(StrategyDiscriminatorALICE,self).__init__(*args,**kwargs)
