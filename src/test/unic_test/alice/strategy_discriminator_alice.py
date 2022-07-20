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
        self.Dyy    = accel(network.DCGAN_Dx( self.opt.config['Dyy'], self.opt,model_name='Dyy')).cuda()
        self.Dzzb   = accel(network.DCGAN_Dz( self.opt.config['Dzzb'],self.opt,model_name='Dzzb')).cuda()

        self.Dx     = accel(network.DCGAN_Dx( self.opt.config['Dx'],  self.opt,model_name='Dx')).cuda()
        self.Dsx    = accel(network.DCGAN_DXZ(self.opt.config['Dsx'], self.opt,model_name='Dsx')).cuda()
        self.Dzf    = accel(network.DCGAN_Dz( self.opt.config['Dzf'], self.opt,model_name='Dzf')).cuda()
        self.Dszf   = accel(network.DCGAN_DXZ(self.opt.config['Dszf'],self.opt,model_name='Dszf')).cuda()
        self.Dxz    = accel(network.DCGAN_DXZ(self.opt.config['Dxz'], self.opt,model_name='Dxz')).cuda()
        self.Dxx    = accel(network.DCGAN_Dx( self.opt.config['Dxx'], self.opt,model_name='Dxx')).cuda()
        self.Dzzf   = accel(network.DCGAN_Dz( self.opt.config['Dzzf'],self.opt,model_name='Dzzf')).cuda()
        

        self._discriminators  = [ self.Dy, self.Dsy, self.Dzb, self.Dszb, self.Dyz, 
                        self.Dx, self.Dsx, self.Dzf, self.Dszf, self.Dxz]
        
        self._name_discriminators = ['Dy','Dsy', 'Dzb', 'Dszb', 'Dyz', 'Dyy', 'Dzzb', 
            'Dx', 'Dsx', 'Dzf', 'Dszf', 'Dxz', 'Dxx', 'Dzzf']
        super(StrategyDiscriminatorALICE,self).__init__(*args,**kwargs)
    
    def _optimizer(self,*args,**kwargs):
        return reset_net(self._discriminators,lr = self.rlr, optim='adam', b1=0.5, b2=0.9999,
            alpha=0.90, weights_decay=self.weight_decay)
    
    def _discriminate_conjoint_yz(self,y,y_gen,z, z_gen,*args,**kwargs):
        Dreal_yz = self.Dyz(zcat(self.Dy(y), self.Dzb(z_gen)))
        Dfake_yz = self.Dyz(zcat(self.Dy(y_gen), self.Dzb(z)))
        return Dreal_yz, Dfake_yz

    def _discriminate_crosss_entropy_zd(self,z, zr,*args,**kwargs):
        Dreal_z = self.Dzzb(zcat(z,z))
        Dfake_z = self.Dzzb(zcat(z,zr))
        return Dreal_z, Dfake_z
    
    def _discriminate_crosss_entropy_y(self,y, yr,*args,**kwargs):
        Dreal_y = self.Dyy(y,y)
        Dfake_y = self.Dyy(y,yr)
        return Dreal_y, Dfake_y

    def _discriminate_conjoint_xz(self,x, x_gen, z,z_gen,*args,**kwargs):
        Dreal_yz = self.Dyz(zcat(self.Dx(x), self.Dzf(z_gen)))
        Dfake_yz = self.Dyz(zcat(self.Dx(x_gen),self.Dzf(z)))
        return Dreal_yz, Dfake_yz
    
    def _discriminate_crosss_entropy_zf(self,z, zr,*args,**kwargs):
        Dreal_z = self.Dzzf(zcat(z,z))
        Dfake_z = self.Dzzf(zcat(z,zr))
        return Dreal_z, Dfake_z
    
    def _discriminate_crosss_entropy_x(self,x, xr,*args,**kwargs):
        Dreal_y = self.Dxx(x,x)
        Dfake_y = self.Dxx(x,xr)
        return Dreal_y, Dfake_y
    
    def _get_discriminators(self,*args,**kwargs):
        return self._discriminators

    def _get_name_discriminators(self):
        return self._name_discriminators

