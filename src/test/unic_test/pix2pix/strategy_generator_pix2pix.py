import torch
from app.agent.unic.strategy_generators import IStrategyGenerator
from common.common_nn import reset_net, zcat
from torch.utils.tensorboard import SummaryWriter

class StrategyGeneratorPix2Pix(IStrategyGenerator):
    def __init__(self,network, accel, opt, *args, **kwargs):
        self.opt = opt
        
        self.elr    = self.opt.config["hparams"]['generators.encoder.lr']
        self.dlr    = self.opt.config["hparams"]['generators.decoder.lr']
        self.weight_decay = self.opt.config["hparams"]['generators.weight_decay']
    
        self.Fxy = accel(network.Encoder(self.opt.config['Fxy'], self.opt,model_name='Fxy')).cuda()
        self.Gxy = accel(network.Decoder(self.opt.config['Gxy'],self.opt,model_name='Gxy')).cuda()

        self._generators = [self.Fxy, self.Gxy]
        self._name_generators = ['Fy', 'Gy']

        super(StrategyGeneratorPix2Pix, self).__init__(*args, **kwargs)

    
    def _optimizer_encoder(self,*args,**kwargs):
        return reset_net([self.Fxy], optim='adam',alpha=0.9,lr=self.elr,b1=0.5,b2=0.999, 
            weight_decay=self.weight_decay)

    def _optimizer_decoder(self,*args, **kwargs):
        return reset_net([self.Gxy],optim='adam',alpha=0.9,lr=self.dlr,b1=0.5,b2=0.999, 
            weights_decay=self.weight_decay)

    def _architecture(self,explore,*args,**kwargs):
        if explore:
            writer_encoder = SummaryWriter(self.opt.config['log_dir']['debug.encoder_writer'])
            writer_decoder = SummaryWriter(self.opt.config['log_dir']['debug.decoder_writer'])
            writer_encoder.add_graph(next(iter(self.Fxy.children())),
                            torch.randn(10,6,4096).cuda())
            writer_decoder.add_graph(next(self.Gxy.children()), 
                            (torch.randn(10,1,512).cuda()))
    
    def _get_generators(self,*args, **kwargs):
        return self._generators

    def _get_name_generators(self, *args, **kwargs):
        return self._name_generators