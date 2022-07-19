import torch
from torch.utils.tensorboard import SummaryWriter
from core.trainer.basic_frame import Agent
from common.common_nn import reset_net, set_weights


class Generators(Agent):
    def __init__(self,network,config,logger, accel,opt, gradients_tracker,debug_writer,*args, **kwargs):
        self.config = config
        self.opt    = opt
        self.generators = []
        self.gradients_tracker = gradients_tracker
        self.debug_writer = debug_writer
        self.glr    = self.opt.config["hparams"]['generators.lr']
        self.weight_decay = self.opt.config["hparams"]['generators.weight_decay']

        self.Fxy    = accel(network.Encoder(self.opt.config['F'], self.opt,model_name='F')).cuda()
        self.Gy     = accel(network.Decoder(self.opt.config['Gy'],self.opt,model_name='Gy')).cuda()

        self.generators= [self.Fxy, self.Gy]
        self.optimizer = reset_net(self.generators,func=set_weights,
                lr=self.glr,b1=0.5,b2=0.9999,weight_decay=self.weight_decay
            )
        self._architecture()
        super(Generators,self).__init__(config, logger, accel,*args, **kwargs)
    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.generators,epoch)
    
    def track_weight(self, epoch):
        self.track_weight_change(writer =  self.debug_writer, tag = 'F[cnn_common]', 
            model= self.Fxy.module.cnn_common.eval(),epoch = epoch)
        self.track_weight_change(writer =  self.debug_writer, tag = 'F[cnn_broadband]', 
            model= self.Fxy.module.cnn_broadband.eval(),epoch = epoch)
        
        self.track_weight_change(writer =  self.debug_writer, tag = 'Gy', 
            model= self.Gy.module.eval(),epoch = epoch)

    def _architecture(self):
        writer_encoder = SummaryWriter(self.opt.config['log_dir']['debug.encoder_writer'])
        writer_decoder = SummaryWriter(self.opt.config['log_dir']['debug.decoder_writer'])
        writer_encoder.add_graph(next(iter(self.Fxy.children())),
                        torch.randn(10,6,4096).cuda())
        writer_decoder.add_graph(next(self.Gy.children()), 
                        (torch.randn(10,4,128).cuda(),torch.randn(10,4,128).cuda()))
