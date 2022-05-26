from core.trainer.basic_frame import Agent
from common.common_nn import reset_net, set_weights,zcat
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

        self.discriminators  = [ self.Dy, self.Dsy, self.Dzb, self.Dzb, self.Dszb, self.Dyz, 
                        self.Dx, self.Dsx, self.Dzf, self.Dszf, self.Dxz]

        self.optimizer = reset_net(self.discriminators,func=set_weights,lr = self.rlr,
                    optim='rmsprop', b1 = 0.5, b2 = 0.9999,
                    weight_decay=self.weight_decay
                )
        super(Discriminators,self).__init__(config, logger, accel,*args, **kwargs)
    
    def track_gradient(self,epoch):
        self.track_gradient_change(self.gradients_tracker,self.discriminators,epoch)
    
    def track_weight(self,epoch):
        for net in self.discriminators:
            self.track_weight_change(writer =  self.debug_writer, tag = net.name, model= net,epoch = epoch)
        
    def discriminate_xz(self,x,xr,z,zr):
        # Discriminate real
        wnx,wnz,*others = noise_generator(x.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzf(zcat(zr,wnz)) #--OK: no batchNorm
        ftx         = self.Dx(zcat(x,wnx)) #--with batchNorm
        zrc         = zcat(ftx,ftz)
        ftxz        = self.Dxz(zrc)   #no batchNorm
        Dxz         = ftxz

        # Discriminate fake
        wnx,wnz,*others = noise_generator(x.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzf(zcat(z,wnz))
        ftx         = self.Dx(zcat(xr,wnx))
        zrc         = zcat(ftx,ftz)
        ftzx        = self.Dxz(zrc)
        Dzx         = ftzx
        return Dxz,Dzx #,ftr,ftf
    
    def discriminate_yz(self,y,yr,z,zr):
        # Discriminate real
        wny,wnz,*others = noise_generator(y.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzb(zcat(zr,wnz)) #--OK : no batchNorm
        ftx         = self.Dy(zcat(y,wny)) # --OK : with batchNorm
        zrc         = zcat(ftx,ftz)
        ftxz        = self.Dyz(zrc)
        Dxz         = ftxz
        
        # Discriminate fake
        wny,wnz,*others = noise_generator(y.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzb(zcat(z,wnz))
        ftx         = self.Dy(zcat(yr,wny))
        zrc         = zcat(ftx,ftz)
        ftzx        = self.Dyz(zrc)
        Dzx         = ftzx
        return Dxz,Dzx 

    def discriminate_marginal_y(self,y,yr):
        # We apply in frist convolution from the y signal ...
        # the we flatten thaf values, a dense layer is added 
        # and a tanh before the output of the signal. This 
        # insure that we have a probability distribution.
        wny,*others = noise_generator(y.shape,yr.shape,app.DEVICE,{'mean':0., 'std':self.std})
        fty         = self.Dy(zcat(y,wny))       
        Dreal       = self.Dsy(fty)

        # Futher more, we do the same but for the reconstruction of the 
        # broadband signals
        wny,*others = noise_generator(y.shape,yr.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftyr        = self.Dy(zcat(yr,wny))
        Dfake       = self.Dsy(ftyr)
        return Dreal, Dfake
    
    def discriminate_marginal_zd(self,z,zr):
        # We apply in first the same neurol network used to extract the z information
        # from the adversarial losses. Then, we extract the sigmoïd afther 
        # the application of flatten layer,  dense layer
        wnz,*others = noise_generator(z.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzb(zcat(z,wnz))
        Dreal       = self.Dszb(ftz)

        # we do the same for reconstructed or generated z
        wnz,*others = noise_generator(z.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzr        = self.Dzb(zcat(zr,wnz))
        Dfake       = self.Dszb(ftzr)
        return Dreal, Dfake

    def discriminate_marginal_x(self,x,xr):
        # We apply a the same neural netowrk used to match the joint distribution
        # and we extract the probability distribution of the signals
        wnx,*others = noise_generator(x.shape,xr.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftx         = self.Dx(zcat(x,wnx))
        Dreal       = self.Dsx(ftx)

        # Doing the same for reconstruction/generation of x
        wnx,*others = noise_generator(x.shape,xr.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftxr        = self.Dx(zcat(xr,wnx))
        Dfake       = self.Dsx(ftxr)
        return Dreal, Dfake
    
    def discriminate_marginal_zxy(self,zxy,zxyr):
        # This function extract the probability of the marginal
        # It's reuse the neural network in the joint probability distribution
        # On one hand, we extract the real values.
        wnz,*others = noise_generator(zxy.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzxy       = self.Dzf(zcat(zxy,wnz))
        Dreal       = self.Dszf(ftzxy)

        wnz,*others = noise_generator(zxy.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzxyr      = self.Dzf(zcat(zxyr,wnz))
        Dfake       = self.Dszf(ftzxyr)
        return Dreal, Dfake

    
    
    
