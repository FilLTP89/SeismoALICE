import torch
from app.agent.unic.strategy_generators import IStrategyGenerator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyGeneratorBiCycleGAN(IStrategyGenerator):
    def __init__(self,network, accel, opt, *args, **kwargs):
        self.opt = opt
        
        self.elr    = self.opt.config["hparams"]['generators.encoder.lr']
        self.dlr    = self.opt.config["hparams"]['generators.decoder.lr']
        self.weight_decay = self.opt.config["hparams"]['generators.weight_decay']
    
        self.Fy = accel(network.Encoder(self.opt.config['Fx'], self.opt,model_name='Fxy')).cuda()
        self.Gxy= accel(network.Encoder(self.opt.config['Gxy'], self.opt,model_name='Gxy')).cuda()
        self._generators = [self.Fy]
        self._name_generators = ['Fxy']

        super(StrategyGeneratorBiCycleGAN, self).__init__(*args, **kwargs)

    def _optimizer_encoder(self,*args,**kwargs):
        return reset_net([self.Fy], optim='adam',alpha=0.9,lr=self.elr,b1=0.5,b2=0.999, weight_decay=self.weight_decay)

    def _optimizer_decoder(self,*args, **kwargs):
        return reset_net([self.Gxy], optim='adam',alpha=0.9,lr=self.elr,b1=0.5,b2=0.999, weight_decay=self.weight_decay)

    def _architecture(self,explore,*args,**kwargs):
        if explore:
            writer_encoder = SummaryWriter(self.opt.config['log_dir']['debug.encoder_writer'])            
            writer_encoder.add_graph(next(iter(self.Fy.children())),
                            torch.randn(10,6,4096).cuda())

    def _get_generators(self,*args, **kwargs):
        return self._generators

    def _get_name_generators(self, *args, **kwargs):
        return self._name_generators
    
    def reparametrization_trick(self,mu, logvar):
        std      = torch.exp(logvar/2)
        sample_z = torch.randn(0, 1, (mu.size(0), 512))
        z = sample_z* std + mu
        return z