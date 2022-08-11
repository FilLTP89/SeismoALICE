import torch
from app.agent.simple.strategy_generators import IStrategyGenerator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyGeneratorALICE(IStrategyGenerator):
    def __init__(self,network, accel, opt, *args, **kwargs):
        self.opt = opt
        
        self.elr    = self.opt.config["hparams"]['generators.encoder.lr']
        self.dlr    = self.opt.config["hparams"]['generators.decoder.lr']
        self.weight_decay = self.opt.config["hparams"]['generators.weight_decay']
        
        self.Fy = accel(network.Encoder(self.opt.config['Fy'], self.opt,model_name='Fy')).cuda()
        self.Gy = accel(network.Decoder(self.opt.config['Gy'],self.opt,model_name='Gy')).cuda()

        self._generators = [self.Fy, self.Gy]
        self._name_generators = ['Fy', 'Gy']

        super(StrategyGeneratorALICE, self).__init__(*args, **kwargs)

    
    def _optimizer_encoder(self,*args,**kwargs):
        return reset_net([self.Fy], optim='adam',alpha=0.9,lr=self.elr,b1=0.5,b2=0.999)

    def _optimizer_decoder(self,*args, **kwargs):
        return reset_net([self.Gy],optim='adam',alpha=0.9,lr=self.dlr,b1=0.5,b2=0.999)

    def _architecture(self,explore,*args,**kwargs):
        if explore:
            writer_encoder = SummaryWriter(self.opt.config['log_dir']['debug.encoder_writer'])
            writer_decoder = SummaryWriter(self.opt.config['log_dir']['debug.decoder_writer'])
            writer_encoder.add_graph(next(iter(self.Fy.children())),torch.randn(10,6,4096).cuda())
            writer_decoder.add_graph(next(self.Gy.children()),(torch.randn(10,1,512).cuda()))
    
    def _get_generators(self,*args, **kwargs):
        return self._generators

    def _get_name_generators(self, *args, **kwargs):
        return self._name_generators