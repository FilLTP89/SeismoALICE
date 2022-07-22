import torch
from app.agent.unic.strategy_generators import IStrategyDiscriminator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyDiscriminatorPix2Pix(IStrategyDiscriminator):
    def __init__(self,network, accel, opt, *args, **kwargs):
        self.opt = opt
        self.elr    = self.opt.config["hparams"]['generators.encoder.lr']
        self.dlr    = self.opt.config["hparams"]['generators.decoder.lr']
        self.weight_decay = self.opt.config["hparams"]['generators.weight_decay']
    
        self.Dxy = accel(network.DCGAN_Dx(self.opt.config['Dxy'], self.opt,model_name='Dsy')).cuda()
        self.Dsz = accel(network.DCGAN_Dz(self.opt.config['Dsz'], self.opt,model_name='Dsz')).cuda()
        self._discriminators = [self.Dsy, self.Dsy]
        self._name_discriminators = ['Dxy','Dsz']
        super(StrategyDiscriminatorPix2Pix,self).__init__(*args,**kwargs)
    
    def _discriminate_conjointe_xy(self,x,y,yr):
        Dreal = self.Dxy(zcat(x,y))
        Dfake = self.Dxy(zcat(x,yr))
        return Dreal,Dfake
    
    def _discriminate_marginal_zd(self,z, zr):
        Dreal = self.Dsz(z)
        Dfake = self.Dsz(zr)
        return Dreal, Dfake
    
    def _architecture(self, explore,*args,**kwargs):
        if explore:
            writer_dxy  = SummaryWriter(self.opt.config['log_dir']['debug.dxy_writer'])
            writer_dsz  = SummaryWriter(self.opt.config['log_dir']['debug.dsz_writer'])

            writer_dxy.add_graph(next(iter(self.Dxy.children())),torch.randn(10,6,4096).cuda())
            writer_dsz.add_graph(next(iter(self.Dsz.children())),torch.randn(10,1,512).cuda())

    def _get_discriminators(self):
        return self._discriminators

    def _get_name_discriminators(self):
        return self._name_discriminators
    
    def _optimizer(self):
        return reset_net(self._discriminators,lr = self.rlr, optim='adam', b1=0.5, b2=0.999, alpha=0.90)