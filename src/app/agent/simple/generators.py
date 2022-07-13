import torch
from torch.utils.tensorboard import SummaryWriter
from core.trainer.basic_frame import Agent
from common.common_nn import reset_net


class Generators(Agent):
    def __init__(self,network,config,logger, accel,opt, gradients_tracker,debug_writer,*args, **kwargs):
        self.config = config
        self.opt    = opt
        self.generators = []
        self.gradients_tracker = gradients_tracker
        self.debug_writer = debug_writer
        self.elr    = self.opt.config["hparams"]['generators.encoder.lr']
        self.dlr    = self.opt.config["hparams"]['generators.decoder.lr']
        self.weight_decay = self.opt.config["hparams"]['generators.weight_decay']
    
        
        self.Fy = accel(network.Encoder(self.opt.config['F'], self.opt,model_name='F')).cuda()
        self.Gy = accel(network.Decoder(self.opt.config['Gy'],self.opt,model_name='Gy')).cuda()
        
        self.generators = [self.Fy, self.Gy]
        self.current_val= 0
        self.optimizer_encoder = reset_net([self.Fy],
                optim='adam',alpha=0.9,lr=self.elr,b1=0.5,b2=0.9999)
        self.optimizer_decoder = reset_net([self.Gy],
                optim='adam',alpha=0.9,lr=self.dlr,b1=0.5,b2=0.9999,weight_decay=self.weight_decay
            )

        self._architecture()
        super(Generators,self).__init__(self.generators, [self.optimizer_encoder,self.optimizer_decoder], 
            config, logger, accel,*args, **kwargs)

    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.generators,epoch)
    
    def track_weight(self, epoch):
        self.track_weight_change(writer =  self.debug_writer, tag = 'F[cnn_common]', 
            model= self.Fy.module.cnn1.eval(),epoch = epoch)
        
        self.track_weight_change(writer =  self.debug_writer, tag = 'Gy', 
            model= self.Gy.module.cnn1.eval(),epoch = epoch)

    def _architecture(self):
        writer_encoder = SummaryWriter(self.opt.config['log_dir']['debug.encoder_writer'])
        writer_decoder = SummaryWriter(self.opt.config['log_dir']['debug.decoder_writer'])
        writer_encoder.add_graph(next(iter(self.Fy.children())),
                        torch.randn(10,6,4096).cuda())
        writer_decoder.add_graph(next(self.Gy.children()), 
                        (torch.randn(10,1,512).cuda()))
