import torch
from core.trainer.basic_frame import Agent
from torch.utils.tensorboard import SummaryWriter
from common.common_nn import reset_net,zcat
from tools.generate_noise import noise_generator
from configuration import app
class Discriminators(Agent):
    def __init__(self,network,config,logger, accel, opt, gradients_tracker,debug_writer,*args, **kwargs):
        self.config = config
        self.opt    = opt
        self.std    = 1.0
        self.debug_writer = debug_writer
        self.gradients_tracker = gradients_tracker
        self.discriminators  = []

        self.rlr = self.opt.config["hparams"]['discriminators.lr']
        self.weight_decay = self.opt.config["hparams"]['discriminators.weight_decay']
        
        # self.Dy     = accel(network.DCGAN_Dx( self.opt.config['Dy'],  self.opt,model_name='Dy')).cuda()
        # self.Dyy    = accel(network.DCGAN_Dx(self.opt.config['Dyy'], self.opt,model_name='Dyy')).cuda()
        # self.Dyz    = accel(network.DCGAN_DXZ(self.opt.config['Dyz'], self.opt,model_name='Dyz')).cuda()
        self.Dsy    = accel(network.DCGAN_Dx(self.opt.config['Dsy'], self.opt,model_name='Dsy')).cuda()
        # self.Dzb    = accel(network.DCGAN_Dz( self.opt.config['Dzb'], self.opt,model_name='Dzb')).cuda()
        # self.Dzzb   = accel(network.DCGAN_Dz(self.opt.config['Dzzb'],self.opt,model_name='Dzzb')).cuda()
        self.Dszb   = accel(network.DCGAN_Dz(self.opt.config['Dszb'],self.opt,model_name='Dszb')).cuda()
        
        self.discriminators  = [self.Dsy,self.Dszb]
        # self.discriminators = [self.Dy, self.Dyy, self.Dyz, self.Dzb, self.Dzzb]

        self.optimizer = reset_net(self.discriminators,lr = self.rlr,
                    optim='adam', b1=0.5, b2=0.9999, alpha=0.90,
                    weight_decay=self.weight_decay)
        self._architecture()
        super(Discriminators,self).__init__(self.discriminators,self.optimizer, config, logger, accel,*args, **kwargs)
    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.discriminators,epoch)
    
    def track_weight(self,epoch):
        for net in self.discriminators:
            self.track_weight_change(writer =  self.debug_writer, tag = net.module.model_name,
            model= net.module.cnn.eval(), epoch = epoch)
    
    def _architecture(self):
        writer_dsy   = SummaryWriter(self.opt.config['log_dir']['debug.dsy_writer'])
        # writer_dyy  = SummaryWriter(self.opt.config['log_dir']['debug.dyy_writer'])
        writer_dszb  = SummaryWriter(self.opt.config['log_dir']['debug.dszb_writer'])
        # writer_dzzb = SummaryWriter(self.opt.config['log_dir']['debug.dzzb_writer'])
        # writer_dyz  = SummaryWriter(self.opt.config['log_dir']['debug.dyz_writer'])

        writer_dsy.add_graph(next(iter(self.Dsy.children())),torch.randn(10,3,4096).cuda())
        # writer_dyy.add_graph(next(iter(self.Dyy.children())),torch.randn(10,6,4096).cuda())
        writer_dszb.add_graph(next(iter(self.Dszb.children())),torch.randn(10,1,512).cuda())
        # writer_dzzb.add_graph(next(iter(self.Dzzb.children())), torch.randn(10,8,128).cuda())
        # writer_dyz.add_graph(next(iter(self.Dyz.children())), torch.randn(10,2,512).cuda())
    
    def discriminate_yz(self,y,yr,z,zr):
        # Discriminate real
        ftz         = self.Dzb(zr) #--OK : no batchNorm
        ftx         = self.Dy(y) # --OK : with batchNorm
        zrc         = zcat(ftx,ftz)
        ftxz        = self.Dyz(zrc)
        Dxz         = ftxz
        
        # Discriminate fake
        ftz         = self.Dzb(z)
        ftx         = self.Dy(yr)
        zrc         = zcat(ftx,ftz)
        ftzx        = self.Dyz(zrc)
        Dzx         = ftzx
        return Dxz,Dzx 
    
    def discriminate_yy(self,y,yr):
        Dreal = self.Dyy(zcat(y,y))
        Dfake = self.Dyy(zcat(y,yr))
        return Dreal, Dfake

    def discriminate_zzb(self,z,zr):
        Dreal = self.Dzzb(zcat(z,z))
        Dfake = self.Dzzb(zcat(z,zr))
        return Dreal, Dfake

    def discriminate_marginal_y(self,y,yr):
        # We apply in frist convolution from the y signal ...
        # the we flatten thaf values, a dense layer is added 
        # and a tanh before the output of the signal. This 
        # insure that we have a probability distribution.
        Dreal       = self.Dsy(y)
        # Futher more, we do the same but for the reconstruction of the 
        # broadband signals
        Dfake       = self.Dsy(yr)
        return Dreal, Dfake
    
    def discriminate_marginal_zd(self,z,zr):
        # We apply in first the same neurol network used to extract the z information
        # from the adversarial losses. Then, we extract the sigmoïd afther 
        # the application of flatten layer,  dense layer
        Dreal       = self.Dszb(z)
        # we do the same for reconstructed or generated z
        Dfake       = self.Dszb(zr)
        return Dreal, Dfake

    
    
    
