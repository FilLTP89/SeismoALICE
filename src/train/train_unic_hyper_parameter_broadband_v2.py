# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from copy import deepcopy
import imp
from re import S
from tabnanny import verbose
import torch
import random
from common.common_nn import *
from common.common_torch import * 
import plot.plot_tools as plt
from tools.generate_noise import noise_generator
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
import time
import GPUtil

from configuration import app
from tqdm import  tqdm,trange
import optuna
from optuna.trial import TrialState
u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

# coder en dure dans le programme 
class trainer(object):
    
    def __init__(self,cv, trial=None, study_dir=None):
        # In this constructor we setup the convolutional neural network for auto-encoder 
        # and for the discriminators. 
        # cv        : [dict] contains the global variable we need for the training 
        # trail     : variable from optuna, this help for hyper parameter tuning of the code
        # study_dir : help for tensorboard catching informations

        super(trainer, self).__init__()

        # attribut we need for the programs

        self.cv         = cv
        self.gr_norm    = []
        self.std        = 1.0
        self.trial      = trial
        self.study_dir  = study_dir
        
        # passing the global variables for the whole program
        globals().update(cv)
        globals().update(opt.__dict__)
        
        # If we need to fine the best hyper parameter for tuning we could use set up thoses 
        # as follow. A distribution value is catch in the memory and the more we launch the 
        # code the more different values are passed. 

        # When trial is not none ...
        # we tune the learning weigh for autp-encoder and discriminators. Values are
        # between [0.0001; 0.001]. Increasing the learning rate help in case of a huge 
        # batch size.
        # We could also extract the weight decay. but make sure you make a grid search. Some 
        # papers say to make small change of that values. decreasing the weight decay help
        # for stability and convergence.
        if self.trial!=None:
            self.glr = self.trial.suggest_float("glrx",0.0004, 0.0005,log=True)
            self.rlr = self.trial.suggest_float("rlrx",0.0001, 0.0002,log=True)
            self.weight_decay = 0.00001 #self.trial.suggest_float("weight_decay",1.E-5,1.E-3,log=True)
        # Else We extract those parameters from the config file. So make sure you get in 
        # the ./config/ folder.  
        else:
            try:
                self.glr = float(opt.config["hparams"]['glry'])
                self.rlr = float(opt.config["hparams"]['rlry'])
                self.weight_decay = 0.00001 # float(opt.config["hparams"]["weight_decay"])
            except Exception as e:
                self.glr = opt.glr
                self.rlr = opt.rlr
                self.weight_decay = 0.00001
                pass

        app.logger.info(f'glr = {self.glr}')
        app.logger.info(f'rlr = {self.rlr}')
        b1              = 0.5
        b2              = 0.9999
        self.strategy   = strategy
        self.opt        = opt
        self.start_epoch= 0  

        nzd = opt.nzd
        ndf = opt.ndf

        # ngpu_use = torch.cuda.device_count()
        # To make sure that all operation are deterministic in that GPU for reproductibility
        # torch.backends.cudnn.deterministic  = True
        # torch.backends.cudnn.benchmark      = False

        self.Dnets      = []
        self.optz       = []
        self.oGyx       = None

        # Some intermediate lossse are stocked here. This values help use see if the program 
        # converges or not.
        self.losses     = {
            # Dloss for ALI and cycle consistency
            'Dloss':[0],'Dloss_ali':[0],'Dloss_ali_y':[0],'Dloss_ali_x':[0],
            # Dloss for marginals
            'Dloss_marginal':[0],'Dloss_marginal_y':[0],
            'Dloss_marginal_zd':[0],'Dloss_marginal_x':[0],'Dloss_marginal_zf':[0],
            # Gloss for ALI and cycle consistency
            #Gloss for marginalss
            'Gloss':[0],'Gloss_ali':[0],'Gloss_ali_x':[0],'Gloss_ali_y':[0],
            'Gloss_marginal':[0],'Gloss_marginal_y':[0],'Gloss_marginal_zd':[0],
            'Gloss_marginal_x':[0],'Gloss_marginal_zf':[0],
            #Gloss for 
            'Gloss_rec':[0],'Gloss_rec_y':[0],'Gloss_rec_x':[0],'Gloss_rec_zd':[0],
            'Gloss_rec_zx':[0],
            'Gloss_rec_zxy':[0],'Gloss_rec_x':[0],'kstestz':[0],
        }

        self.gradients = {
            'Fxy':[0],'Gy':[0],
            'Dy':[0],'Dx':[0],'Dsy':[0],'Dsx':[0],'Dzb':[0], 'Dszb':[0],
            'Dyz':[0],'Dxz':[0],'Dzf':[0],'Dszf':[0]
        }
        
        # In case of tensorboard we want to see the intermediate inputs.
        if self.trial == None:
            self.writer_train = SummaryWriter(f'{opt.summary_dir}/training')
            self.writer_val   = SummaryWriter(f'{opt.summary_dir}/validation')
            self.writer_debug = SummaryWriter(f'{opt.summary_dir}/debug')
        else:
            hparams_dir         = f'{self.study_dir}/hparams/'
            self.writer_hparams = SummaryWriter(f'{hparams_dir}')
            # self.writer_hparams_graph_encoder = SummaryWriter(f'{hparams_dir}/graph/encoder')
            # self.writer_hparams_graph_decoder = SummaryWriter(f'{hparams_dir}/graph/decoder')

        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        net = Network(DataParalleleFactory())

        self.trn_loader, self.vld_loader, self.tst_loader = trn_loader, vld_loader, tst_loader

        if 'unique' in t:
            self.style='ALICE'
            # act = acts[self.style]
            n = self.strategy['unique']

            self.Fxy  = net.Encoder(opt.config['F'],  opt)
            self.Gy  = net.Decoder(opt.config['Gy'], opt)
            # self.Gx  = net.Decoder(opt.config['Gx'], opt)
            self.Fxy  = nn.DataParallel(self.Fxy).cuda()
            self.Gy  = nn.DataParallel(self.Gy).cuda()
            # self.Gx  = nn.DataParallel(self.Gx).cuda()

            if  self.strategy['tract']['unique']:
                if None in n:       
                    self.FGf  = [self.Fxy,self.Gy]
                    self.oGyx = reset_net(
                        self.FGf,
                        func=set_weights,
                        lr=self.glr,b1=b1,b2=b2,
                        weight_decay=0.00001
                    )

                    # self.g_scheduler = MultiStepLR(self.oGyx,milestones=[30,80], gamma=0.1) 
                    # self.oGyx = Adam(ittc(self.F_.branch_common.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,betas=(b1,b2),
                    #     weight_decay=0.00001)

                    # self.oGy = Adam(ittc(self.F_.branch_broadband.parameters(),
                    #     self.Gy.parameters()),
                    #     lr=glr,betas=(b1,b2),
                    #     weight_decay=0.00001)
                else: 
                    # breakpoint()
                    self.Fxy.load_state_dict(tload(n[0])['model_state_dict'])
                    # self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    # self.Gx.load_state_dict(tload(n[2])['model_state_dict'])

                    app.logger.info("considering master part as pretrained : no grad required")
                    for param in self.Fxy.module.master.parameters():
                        param.requires_grad = False

                    app.logger.info("considering common part as pretrained : no grad required")
                    for param in self.Fxy.module.cnn_common.parameters():
                        param.requires_grad = False

                    self.oGyx = MultiStepLR(Adam(ittc(self.Fxy.parameters()),
                        lr=self.glr,betas=(b1,b2),
                        weight_decay=self.weight_decay))
                    
                    # self.g_scheduler = MultiStepLR(self.oGyx,milestones=[30,80], gamma=0.1)
                    # self.oGy = Adam(ittc(self.F_.branch_broadband.parameters(),
                    #     self.Gy.parameters()),
                    #     lr=ngpu_use*glr,betas=(b1,b2),
                    #     weight_decay=0.00001)
                    
                    self.FGf  = [self.Fxy,self.Gy]
                    app.logger.info("intialization of Gy and the broadband branch for the broadband training ...")
                    self.oGyx = reset_net([self.Gy,self.Fxy.module.cnn_broadband],
                        func=set_weights,lr=self.glr,b1=b1,b2=b2,
                        weight_decay=self.weight_decay)
                    # self.g_scheduler = MultiStepLR(self.oGyx,milestones=[30,80], gamma=0.1)

                    # self.oGyx = RMSProp(ittc(self.F_.parameters(),
                    #     self.Gy.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,alpha=b2,
                    #     weight_decay=0.00001)
                self.optz.append(self.oGyx)
                # self.optz.append(self.oGy)
                breakpoint()
                self.Dy     = net.DCGAN_Dx( opt.config['Dy'],  opt)
                self.Dsy    = net.DCGAN_DXZ(opt.config['Dsy'],  opt)
                self.Dzb    = net.DCGAN_Dz( opt.config['Dzb'], opt)
                self.Dszb   = net.DCGAN_DXZ(opt.config['Dszb'], opt)
                self.Dyz    = net.DCGAN_DXZ(opt.config['Dyz'], opt)
                # self.Dzzb   = net.DCGAN_Dz( opt.config['Dzzb'],opt)
                # self.Dyy    = net.DCGAN_Dx( opt.config['Dyy'], opt)

                # self.Dzyx   = net.DCGAN_Dz(opt.config['Dzyx'],opt)
                # self.Dzyy   = net.DCGAN_Dz(opt.config['Dzyy'],opt)
                
                self.Dx     = net.DCGAN_Dx(opt.config['Dx'],  opt)
                self.Dsx    = net.DCGAN_DXZ(opt.config['Dsx'],opt)
                self.Dzf    = net.DCGAN_Dz(opt.config['Dzf'], opt)
                self.Dszf   = net.DCGAN_DXZ(opt.config['Dszf'],opt)
                self.Dxz    = net.DCGAN_DXZ(opt.config['Dxz'],opt)
                # self.Dzzf   = net.DCGAN_Dz(opt.config['Dzzf'],opt)
                # self.Dxx    = net.DCGAN_Dx(opt.config['Dxx'], opt)


                self.Dy     = nn.DataParallel(self.Dy  ).cuda()
                self.Dsy    = nn.DataParallel(self.Dsy  ).cuda()
                self.Dzb    = nn.DataParallel(self.Dzb ).cuda()
                self.Dszb   = nn.DataParallel(self.Dszb ).cuda()
                self.Dyz    = nn.DataParallel(self.Dyz ).cuda()
                # self.Dzzb   = nn.DataParallel(self.Dzzb).cuda()
                # self.Dyy    = nn.DataParallel(self.Dyy ).cuda()
                # self.Dzyx   = nn.DataParallel(self.Dzyx ).cuda()
                # self.Dzyy   = nn.DataParallel(self.Dzyy ).cuda()
                # self.Dxx    = nn.DataParallel(self.Dxx ).cuda()
                # self.Dzzf   = nn.DataParallel(self.Dzzf).cuda()
                self.Dzf    = nn.DataParallel(self.Dzf).cuda()
                self.Dszf   = nn.DataParallel(self.Dszf).cuda()
                self.Dx     = nn.DataParallel(self.Dx).cuda()
                self.Dsx    = nn.DataParallel(self.Dsx).cuda()
                self.Dxz    = nn.DataParallel(self.Dxz).cuda()
                

                self.Dnets.append(self.Dy)
                self.Dnets.append(self.Dsy)
                self.Dnets.append(self.Dzb)
                self.Dnets.append(self.Dszb)
                self.Dnets.append(self.Dyz)
                # self.Dnets.append(self.Dzzb)
                # self.Dnets.append(self.Dyy)

                # self.Dnets.append(self.Dzyx)
                # self.Dnets.append(self.Dzyy)
                # self.Dnets.append(self.Dxx)
                # self.Dnets.append(self.Dzzf)

                self.Dnets.append(self.Dzf)
                self.Dnets.append(self.Dszf)
                self.Dnets.append(self.Dx)
                self.Dnets.append(self.Dsx)
                self.Dnets.append(self.Dxz)

                self.oDyxz = reset_net(
                    self.Dnets,
                    func=set_weights,lr = self.rlr,
                    optim='Adam', b1 = b1, b2 = b2,
                    weight_decay=0.
                )
                
                # self.d_scheduler = MultiStepLR(self.oDyxz,milestones=[30,80], gamma=0.1)

                self.optz.append(self.oDyxz)

            else:
                if None not in n:
                    checkpoint          = tload(n[0])
                    self.start_epoch    = checkpoint['epoch']
                    self.losses         = checkpoint['loss']
                    self.Fxy.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    self.FGf  = [self.Fxy,self.Gy]
                    # self.Gx.load_state_dict(tload(n[2])['model_state_dict'])  
                else:
                    flagF=False

        # self.writer_debug_encoder.add_graph(next(iter(self.F_.children())),torch.randn(128,6,4096).cuda())
        # self.writer_hparams_graph_encoder.add_graph(next(iter(self.F_.children())),torch.randn(10,6,4096).cuda())
        # self.writer_hparams_graph_decoder.add_graph(next(iter(self.Gy.children())), (torch.randn(10,4,128).cuda(),torch.randn(10,4,128).cuda()))
        self.bce_loss = BCE(reduction='mean')
        # breakpoint()
        print("Parameters of  Encoder/Decoders ")
        count_parameters(self.FGf)
        print(self.oGyx)

        print("Parameters of Discriminators ")
        count_parameters(self.Dnets)
        print(self.oDyxz)

        if self.trial == None:
            app.logger.info(f" Root checkpoint: {opt.root_checkpoint}")
            app.logger.info(f" Summary dir    : {opt.summary_dir}")
        else: 
            app.logger.info(f" Tuner dir      : {self.study_dir}")
        app.logger.info(f" Batch size per GPU: {opt.batchSize // torch.cuda.device_count()}")
        
       
    def discriminate_xz(self,x,xr,z,zr):
        # Discriminate real
        
        wnx,wnz,*others = noise_generator(x.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzf(zcat(zr,wnz)) #--OK: no batchNorm
        ftx         = self.Dx(zcat(x,wnx)) #--with batchNorm
        zrc         = zcat(ftx,ftz)
        wnx,*others = noise_generator(zrc.shape,zrc.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftxz        = self.Dxz(zcat(zrc,wnx))   #no batchNorm
        Dxz         = ftxz

        # Discriminate fake
        wnx,wnz,*others = noise_generator(x.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzf(zcat(z,wnz))
        ftx         = self.Dx(zcat(xr,wnx))
        zrc         = zcat(ftx,ftz)
        wnz,*others = noise_generator(zrc.shape,zrc.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzx        = self.Dxz(zcat(zrc,wnz))
        Dzx         = ftzx

        return Dxz,Dzx #,ftr,ftf
    
    def discriminate_yz(self,y,yr,z,zr):
        # Discriminate real
        
        wny,wnz,*others = noise_generator(y.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzb(zcat(zr,wnz)) #--OK : no batchNorm
        ftx         = self.Dy(zcat(y,wny)) # --OK : with batchNorm
        zrc         = zcat(ftx,ftz)
        wnz,*others = noise_generator(zrc.shape,zrc.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftxz        = self.Dyz(zcat(zrc,wnz))
        Dxz         = ftxz
        
        # Discriminate fake
        wny,wnz,*others = noise_generator(y.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzb(zcat(z,wnz))
        ftx         = self.Dy(zcat(yr,wny))
        zrc         = zcat(ftx,ftz)
        wnz,*others = noise_generator(zrc.shape,zrc.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzx        = self.Dyz(zcat(zrc,wnz))
        Dzx         = ftzx

        return Dxz,Dzx 

    def discriminate_marginal_y(self,y,yr):
        # We apply in frist convolution from the y signal ...
        # the we flatten thaf values, a dense layer is added 
        # and a tanh before the output of the signal. This 
        # insure that we have a probability distribution.
       
        wny,*others = noise_generator(y.shape,y.shape,app.DEVICE,{'mean':0., 'std':self.std})
        fty         = self.Dy(zcat(y,wny))
        wny,*others = noise_generator(fty.shape,fty.shape,app.DEVICE,{'mean':0., 'std':self.std})       
        Dreal       = self.Dsy(zcat(fty,wny))
        # Futher more, we do the same but for the reconstruction of the 
        # broadband signals
        wny,*others = noise_generator(y.shape,y.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftyr        = self.Dy(zcat(yr,wny))
        wny,*others = noise_generator(fty.shape,fty.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dfake       = self.Dsy(zcat(ftyr,wny)) 
        return Dreal, Dfake
    
    def discriminate_marginal_zd(self,z,zr):
        # We apply in first the same neurol network used to extract the z information
        # from the adversarial losses. Then, we extract the sigmoïd afther 
        # the application of flatten layer,  dense layer
        
        wnz,*others = noise_generator(z.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftz         = self.Dzb(zcat(z,wnz))
        wnz,*others = noise_generator(ftz.shape,ftz.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dreal       = self.Dszb(zcat(ftz,wnz))
        # we do the same for reconstructed or generated z
        wnz,*others = noise_generator(z.shape,z.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzr        = self.Dzb(zcat(zr,wnz))
        wnz,*others = noise_generator(ftz.shape,ftz.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dfake       = self.Dszb(zcat(ftzr,wnz))
        return Dreal, Dfake

    def discriminate_marginal_x(self,x,xr):
        # We apply a the same neural netowrk used to match the joint distribution
        # and we extract the probability distribution of the signals
        
        wnx,*others = noise_generator(x.shape,x.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftx         = self.Dx(zcat(x,wnx))
        wnz,*others = noise_generator(ftx.shape,ftx.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dreal       = self.Dsx(zcat(ftx,wnz))

        # Doing the same for reconstruction/generation of x
        wnx,*others = noise_generator(x.shape,x.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftxr        = self.Dx(zcat(xr,wnx))
        wnz,*others = noise_generator(ftx.shape,ftx.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dfake       = self.Dsx(zcat(ftxr,wnz))
        return Dreal, Dfake
    
    def discriminate_marginal_zxy(self,zxy,zxyr):
        # This function extract the probability of the marginal
        # It's reuse the neural network in the joint probability distribution
        # On one hand, we extract the real values.
        
        wnz,*others = noise_generator(zxy.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std}) 
        ftzxy       = self.Dzf(zcat(zxy,wnz))
        wnz,*others = noise_generator(ftzxy.shape,ftzxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dreal       = self.Dszf(zcat(ftzxy,wnz))

        # On the other hand, we extract the probability of the fake values
        wnz,*others = noise_generator(zxy.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        ftzxyr      = self.Dzf(zcat(zxyr,wnz))
        wnz,*others = noise_generator(ftzxy.shape,ftzxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Dfake       = self.Dszf(zcat(ftzxyr,wnz))
        return Dreal, Dfake

    # def discriminate_xx(self,x,xr):
    #     # x and xr should have the same distribution !
    #     Dreal = self.Dxx(zcat(x,x))#with batchNorm
    #     Dfake = self.Dxx(zcat(x,xr))
    #     return Dreal,Dfake
    
    # def discriminate_yy(self,y,yr):
    #     Dreal = self.Dyy(zcat(y,y )) #with batchNorm
    #     Dfake = self.Dyy(zcat(y,yr))
    #     return Dreal,Dfake

    # def discriminate_zzb(self,z,zr):
    #     Dreal = self.Dzzb(zcat(z,z )) #no batchNorm
    #     Dfake = self.Dzzb(zcat(z,zr))
    #     return Dreal,Dfake

    # def discriminate_zzf(self,z,zr):
    #     Dreal = self.Dzzf(zcat(z,z )) #no batchNorm
    #     Dfake = self.Dzzf(zcat(z,zr))
    #     return Dreal,Dfake

    # @profile
    def alice_train_discriminator_adv(self,y,zyy,zxy, x):
        # This functions is training the discriminators.
        # We calculate the ALICE + marginal for y and x
        # The loss ALICE is the sum of the loss ALI and CE loss

        # y     : broadband signal size [batch, 3, 4096]
        # x     : low frequency signal size [batch, 3, 4096]
        # zxy   : latent variable that should catching the low frequency. 
        #         It pdf is gaussian
        # zy    : guassian latent variable that should catching the high frequency part
         
        # Preparing for the training. first we put the gradient to zero. Also we put the 
        # auto-encoder (Encoder + Decoder) in eval, based from the ALICE paper. Then we put 
        # all the discriminators in trainig mode.
    
        zerograd(self.optz)
        modalite(self.FGf,  mode = 'eval')
        modalite(self.Dnets,mode = 'train')

        # The First part compute Discriminator losses for the input Y called as broadband 
        # We try to extract form the signal what is the distribution of LF, [0 -1]Hz.
        # We try also to extrac form the signal what is the distributiion of HF [0 -30]Hz. 
        # So in a gaussian space hidden variable zxy is related to LF 
        # and zy is related to HF part

        # Part I.- Training of the Broadband signal
      
        # Before training the noise is added to the input signal. The latent variable is 
        # also concatenated for the training of z. 
        # So What we plan to do is
        # y     -> F(y)     -> G(F(y))
        # zxy,zy-> F(zxy,zy)-> G(F(zxy,zy))
        # So a ALI will evaluate the joint distribution, i.e (y, F(y))~(G(z),z)
        # The cycle consistency will enforce that the y (resp. z) match the correct one,
        # bijectivity.
        # More than that, to make sure that distribution is correct we do the same with
        # the marginal distribution

        # 1.1 We Generate conditional samples
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        zd_inp      = zcat(zxy,zyy)
        y_inp       = zcat(y,wny) 
        # First we generate the yr = G(z) and the zr = F(y) the z represents the concatenation
        # of zxy and zy. 
        y_gen       = self.Gy(zxy,zyy)
        zyy_F,zyx_F,*other = self.Fxy(y_inp)
        zd_gen      = zcat(zyx_F,zyy_F)

        # 1.2 Let's match the proper joint distributions
        # The real part is Dreal_yz  = Dyz(Dy(y),Dzb(F(y))) 
        # The fake part is Dfake_zy  = Dyz(Dy(G(z)),Dzb(z)) 
        Dreal_yz,Dfake_yz = self.discriminate_yz(y,y_gen,zd_inp,zd_gen)
        # Matching of the joint probability distribution (x,F(x)) and (G(z),z)
        # The equation to be satisfied is :
        #       max -(E[log(Dyz(y,F(y)))] + E[log(1 - Dyz(G(z),z))])
        #       REM : BCE(xn,yn) =  -wn*(yn.log(xn) + (1-yn)log(1-xn))
        Dloss_ali_y = self.bce_loss(Dreal_yz,o1l(Dreal_yz))+\
                        self.bce_loss(Dfake_yz,o0l(Dfake_yz))

        # 1.3. We comput the marginal probability distributions
        # The objectif of this part is the calculation of the marginal probability 
        # distribution to enforce loss reconstruction.
        # So we do the evaluation the marginal probability distribution to match 
        # probabilit distribution of y. 
        # The equation to be statisfied is :
        #       -(E[log(Dy(y))] + E[log(1 - Dy(G(z)))])  
        Dreal_y,Dfake_y  = self.discriminate_marginal_y(y,y_gen)
        Dloss_marginal_y = self.bce_loss(Dreal_y,o1l(Dreal_y))+\
                             self.bce_loss(Dfake_y,o0l(Dfake_y))
        # And also, we do the evaluation the marginal probabiliti distribution of 
        # z (ensure it's guassian). 
        # The equation to be satisfied is :
        #       -(E[log(Dzd(zd)] + E[log(1 - Dzd(F(x)])
        Dreal_zd,Dfake_zd= self.discriminate_marginal_zd(zd_inp,zd_gen)
        Dloss_marginal_zd= self.bce_loss(Dreal_zd,o1l(Dreal_zd))+\
                         self.bce_loss(Dfake_zd,o0l(Dfake_zd))

        # Part II.- Training the Filtered signal
        # As before, we prepare the input for the traing. But here, We only pay attention to zxy
        # because we want to force it to capture the low frequency informations that we needed. 
        # So here we do a little change:
        # x         ->  (F|(x)_zxy, 0)  -> G(F|(x)_zxy,0)
        # (zxy, 0)  ->  G(zxy, 0)       -> F(G(zxy, 0))

        # 1.1 Let's compute the Generate samples
        wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        x_inp       = zcat(x,wnx)
        _x_gen      = self.Gy(zxy,o0l(zyy))
        _, zxy_gen, *others = self.Fxy(x_inp)

        # 1.2 Now, we match the probability distribution of (x,F|(x)_zxy) ~ (G(zxy,0), zxy)
        # Because it's easy for the discriminator to view that a distribution is not  a 
        # zero space for zy so we only Discriminate on zxy. 
        # Then the equation we evaluate is :
        #   -(E[log(Dxz(x,F|(x)_zxy))] +  E[log(1 - Dxz(G(zxy,0),zxy))])
        Dreal_xz,Dfake_xz    = self.discriminate_xz(x,_x_gen,zxy,zxy_gen)
        Dloss_ali_x          = self.bce_loss(Dreal_xz,o1l(Dreal_xz))+\
                                self.bce_loss(Dfake_xz,o0l(Dfake_xz)) 

        # 1.3 It is important to evaluate the marginal probability distribution. 
        # For x we should satisfy this equation :
        #   -(E[log(Dx(x,x))] + E[log(Dx(x,G(F|(x)_zxy,0)))])
        Dreal_x,Dfake_x      = self.discriminate_marginal_x(x,_x_gen)
        Dloss_marginal_x     = self.bce_loss(Dreal_x,o1l(Dreal_x))+\
                                self.bce_loss(Dfake_x,o0l(Dfake_x))
        # For zxy, we should satisfy this equation : 
        #   -(E[log(Dzxy(zxy, zxy))] +  E[log(Dzxy(zxy,F(G(zxy,0))))])
        # [TODO] Issues in this marginal to solve. the value decrease too much
        Dreal_zf,Dfake_zf   = self.discriminate_marginal_zxy(zxy,zxy_gen)
        Dloss_marginal_zf   = self.bce_loss(Dreal_zf,o1l(Dreal_zf))+\
                                self.bce_loss(Dfake_x,o0l(Dfake_zf))
        Dloss_marginal      = (
                                Dloss_marginal_y+
                                Dloss_marginal_zd+
                                Dloss_marginal_x+
                                Dloss_marginal_zf
        )
        # ALI losses
        Dloss_ali           = (
                                Dloss_ali_y + 
                                Dloss_ali_x 
                            )
        # Total losses
        Dloss               = Dloss_ali + Dloss_marginal

        Dloss.backward()
        self.oDyxz.step()
        self.gradients['Dy' ].append(self.track_gradient_change(self.Dy))
        self.gradients['Dsy'].append(self.track_gradient_change(self.Dsy))
        self.gradients['Dzb'].append(self.track_gradient_change(self.Dzb))
        self.gradients['Dyz'].append(self.track_gradient_change(self.Dyz))

        self.gradients['Dszb'].append(self.track_gradient_change(self.Dszb))
        self.gradients['Dx' ].append(self.track_gradient_change(self.Dx))
        self.gradients['Dxz'].append(self.track_gradient_change(self.Dxz))
        self.gradients['Dsx'].append(self.track_gradient_change(self.Dsx))
        self.gradients['Dszf'].append(self.track_gradient_change(self.Dszf))
        self.gradients['Dzf'].append(self.track_gradient_change(self.Dzf))

        zerograd(self.optz)
        # clipweights(self.Dnets)
        
        self.losses['Dloss'         ].append(Dloss.tolist())
        self.losses['Dloss_ali'     ].append(Dloss_ali.tolist())
        self.losses['Dloss_ali_y'   ].append(Dloss_ali_y.tolist())
        self.losses['Dloss_ali_x'   ].append(Dloss_ali_x.tolist())

        self.losses['Dloss_marginal'    ].append(Dloss_marginal.tolist())
        self.losses['Dloss_marginal_y'  ].append(Dloss_marginal_y.tolist())
        self.losses['Dloss_marginal_zd' ].append(Dloss_marginal_zd.tolist())
        self.losses['Dloss_marginal_x'  ].append(Dloss_marginal_x.tolist())
        self.losses['Dloss_marginal_zf' ].append(Dloss_marginal_zf.tolist())

    # @profile
    def alice_train_generator_adv(self,y,zyy, zxy, x, epoch, trial_writer=None):
        # This functions is training the auto-encoder (encoder + decoder). But, we continue
        # to train the discriminators for the traing, according the ALICE algorithe. As 
        # we do in the alice_train_discriminator, we will aslo calculate the marginals. 

        # y     : broadband signal size [batch, 3, 4096]
        # x     : low frequency signal size [batch, 3, 4096]
        # zxy   : latent variable that should catching the low frequency. 
        #         It pdf is gaussian
        # zy    : guassian latent variable that should catching the high frequency part
        # epoch : we pass the epoch, this is needed for tensorboard
        # trial_writer  : as the previous variable we get the distribution values of zxy, zy

        # To prepare the the training. First all gradient is set as zero. The auto-encoder 
        # is setted at training mode. The discriminator is aslo at training mode
        zerograd(self.optz)
        modalite(self.FGf,   mode ='train')
        modalite(self.Dnets, mode ='train')
        
        # As we said before the Goal of this function is to compute the loss of Y and the 
        # loss of X. We want to make sure the generator are able to create fake signal as good 
        # as possible to fool the discriminators. This part is a bit different of the 
        # discriminators traing because it's not take some of the equation, according to ALICE paper

        # Part I.- Training of the Broadband signal
        # We add noise to broadband signal. we concatenate z.

        # 1.1 We generate conditional samples.
        # The values G(zxy, zy) and F(y) will be computed. Thoses values will be useful
        # to match joint distributions, and also marginal distribuitions.  

        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        y_inp   = zcat(y,wny)
        zd_inp  = zcat(zxy,zyy)
        y_gen   = self.Gy(zxy,zyy)
        zyy_gen,zyx_gen,*others = self.Fxy(y_inp)
        zd_gen= zcat(zyx_gen,zyy_gen)

        # So, let's evaluate the loss of ALI 
        Dreal_yz,Dfake_yz = self.discriminate_yz(y,y_gen,zd_inp,zd_gen)  
        # There are to way to compute that loss. 
        # On the one hand, The first way is to use the WGAN.
        # The equation is as fallow :   
        #       -(E[D(y,F(y))] -  E[D(G(z),z)]) >= -1
        # Since pytorch are only able to minimize function. A treaky tranformation of 
        # that equation is made But make sure to use clipweight, to avoid gradient 
        # explosure.
        # On the other hand,  we could use ALICE, but we change the sign of that equation, 
        # because we want to minimize the autoencoder. The equation is as follow :
        #       min (E[log(Dyz(y,F(y)))] + E[log(1 - Dyz(G(z),z))])
        #       REM : the BCE loss function  has already added the " - " in the calculation.
        Gloss_ali_y =  self.bce_loss(Dreal_yz,o0l(Dreal_yz))+\
                        self.bce_loss(Dfake_yz,o1l(Dfake_yz))

        # Since, it's hard for the marginal distribution to get good saddle piont and a high 
        # complexe place, the ALI loss will hardly find the good solution. To help in this case, 
        # we compute the marginal on y and z.
        # The marginal loss on y is as follow :  
        #       min (E[log(1 - Dy(G(z)))])
        _ , Dfake_y = self.discriminate_marginal_y(y,y_gen)
        Gloss_marginal_y = (self.bce_loss(Dfake_y,o1l(Dfake_y)))
        # The marginal loss on zd is as follow : 
        #       min (E[log(1 - Dzd(F(x)])
        _, Dfake_zd = self.discriminate_marginal_zd(zd_inp,zd_gen)
        Gloss_marginal_zd= (self.bce_loss(Dfake_zd,o1l(Dfake_zd)))

        # 2. Let's generate the reconstructions, i.e G(F(y)) and F(G(z)). This calulation will 
        # be necessary to cycle consistency loss and also here for the reconstruction losses.
        # By doing this we will ensore that input match to the correcto output and also that 
        # values are as close as possible. 

        # So, we pepare our input for the training, adding noise on broadband, and concatenation
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        # Then we generate reconstructions ...
        y_rec = self.Gy(zyx_gen,zyy_gen)
        y_gen = zcat(y_gen,wny)
        zyy_rec,zyx_rec,*other = self.Fxy(y_gen)
        zd_rec  = zcat(zyx_rec,zyy_rec)
    
        # Let's compute now the cycle consistency losses. The couple (y,y) will be close
        # to (y,G(F(z))), if the following equation is satisfied
        #       E[log(1 - Dxx(y,G(F(y))))]
        # A L1 norm is help full; so is computed by:
        #       || x - G(F(z))      ||
        # _,Dfake_yy      = self.discriminate_yy(y,y_rec)
        # Gloss_cycle_y   = -self.bce_loss(Dfake_yy,o0l(Dfake_yy)) 
        Gloss_rec_y     = torch.mean(torch.abs(y-y_rec))
        # In for the z we have tpo satisfy : 
        #       E[log(1-Dzyy(y, G(F(y))))]
        #       || z - F(G(y))      ||
        # _,Dfake_zzd     = self.discriminate_zzb(zd_inp,zd_rec)
        # Gloss_cycle_zd  = -self.bce_loss(Dfake_zzd,o0l(Dfake_zzd))
        Gloss_rec_zd    = torch.mean(torch.abs(zd_inp-zd_rec))

        # Part II.- Training the Filtered signals
        # In this part we traing the auto encoder to try to generate fake filtered signal 
        # to full the discriminator. So we do : 
        # x         ->  (F|(x)_zxy, 0)  -> G(F|(x)_zxy,0)
        # (zxy, 0)  ->  G(zxy, 0)       -> F(G(zxy, 0))
        # More than that have to make sur that every zxy are indepedant from the zy. Another
        # graph is added then:
        # (zxy,N(0,1))->  G(zxy,N(0,1)) -> F(G(zxy,N(0,1)))

        # As usual, before we train, noising and concatanations. Let's compute the sample by 
        # generating the F|(x)_zxy and G(x).
        nch,nz      = 4, 128
        wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        # wnx_fake,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        # wn          = torch.empty([x.shape[0],nch,nz]).normal_(**app.RNDM_ARGS).to(app.DEVICE)

        x_inp       = zcat(x,wnx)
        zx_gen, zxy_gen, *others = self.Fxy(x_inp)

        _x_gen      = self.Gy(zxy,o0l(zyy))
        #<<<< Test no fake generations to see ...
        #<<<< _x_gen_fake = self.Gy(zxy,wn.detach())

        # We are able to match joint probability distribution and compute losses of marginal
        # probabilities that we need. As we have done in the first part, we should make some 
        # change in the equation for adapt it pytorch.
        # So the equations that we need to satisfy :
        #       min (E[log(Dxz(x,F|(x)_zxy))] + E[log(1 - Dxz(G(zxy),zxy))])
        Dreal_xz,Dfake_xz     = self.discriminate_xz(x,_x_gen,zxy,zxy_gen)
        Gloss_ali_x = self.bce_loss(Dreal_xz,o0l(Dreal_xz))+\
                                self.bce_loss(Dfake_xz,o1l(Dfake_xz)) 
        # Now we compute the loss on the marginal of x. What we want is to satisfy:
        #       min E[log(Dx(x,G(F|(x)_zxy,0)))]
        
        # [TODO] Issues in this marginal to solve. the value decrease too much
        _ , Dfake_x = self.discriminate_marginal_x(x,_x_gen)
        Gloss_marginal_x = (self.bce_loss(Dfake_x,o1l(Dfake_x)))
        # We compute the loss on the marginal of zxy. The equation to be satisfied is: 
        #       min E[log(Dzxy(zxy,F(G(zxy,0))))

        # [TODO] Issues in this marginal to solve. the value decrease too much
        _ ,Dfake_zf = self.discriminate_marginal_zxy(zxy, zxy_gen)
        Gloss_marginal_zf= (self.bce_loss(Dfake_zf,o1l(Dfake_zf)))

        # 2. This second time we generate the reconstuction G(F|(x)_zxy,0) and F|(G(x))_zxy
        # We add noise to x and we concatenate z as cat(zxy,0), because we dont want x to 
        # get "high frequency" information on it
        x_rec       = self.Gy(zxy_gen, o0l(zx_gen))
        x_gen       = zcat(_x_gen,wnx)

        #<<<< Test to see what happens ...
        #<<<< x_gen_fake  = zcat(_x_gen_fake,wnx_fake)

        zxx_rec, zxy_rec, *others = self.Fxy(x_gen)
        #<<<< Test
        #<<<< _, zxy_rec_fake, *others  = self.Fxy(x_gen_fake)
        # zxy_fake    =  zxy_rec_fake
        # Now we wille be able te compute the cycle consistency losses we needed for the training 
        # First,  between te couple (zxy, zxy) and (zxy, F(G(zxy,0)). The equation to be satisfied 
        # is, as follow : 
        #       E[log(1 - Dzzf(zxy,F(G(zxy,0))))]
        # _, Dfake_zf          = self.discriminate_zzf(zxy, zxy_rec)
        # Gloss_cycle_zxy      = -self.bce_loss(Dfake_zf, o0l(Dfake_zf))
        # To insure indepenace of zxy fro zy we do that in the L1 loss. 
        #       || zxy  - F(G(zxy,N(0,I)))  ||
        Gloss_rec_zxy        = torch.mean(torch.abs(zxy - zxy_rec))
        # Secondly, for to match (x,x) and (x, G(F|(x)_zxy,0)) . So the equation that we need to 
        # compute is :
        #       E[log(1-Dxx(x,G(F|(x)_zxy,0)))]
        #       || x    - G(F|(zxy),0)      ||
        # _, Dfake_x           = self.discriminate_xx(x,x_rec)
        # Gloss_cycle_x        = -self.bce_loss(Dfake_x, o0l(Dfake_x))
        Gloss_rec_x          = torch.mean(torch.abs(x - x_rec))
        # This loss is 0 of HF of x signal
        #       || F|(G(zxy,0))_zxx         ||
        Gloss_rec_zx         = torch.mean(torch.abs(zxx_rec))
        
        # 8. Total Loss
        # Gloss_cycle =(
        #                 Gloss_cycle_y + 
        #                 Gloss_cycle_zd +
        #                 Gloss_cycle_x +
        #                 Gloss_cycle_zxy
        #             )
        
        Gloss_marginal = (
                        Gloss_marginal_y+
                        Gloss_marginal_zd+
                        Gloss_marginal_x+
                        Gloss_marginal_zf
        )
        Gloss_rec   =( 
                        Gloss_rec_y +
                        Gloss_rec_zd + 
                        Gloss_rec_zx +
                        Gloss_rec_x +
                        Gloss_rec_zxy 
                    )
        Gloss_ali   = (
                        Gloss_ali_y+
                        Gloss_ali_x
                    )
        Gloss       = (
                        Gloss_ali+
                        Gloss_marginal+ 
                        Gloss_rec
                    )   

        if epoch%25 == 0:
            _, pvalue = is_gaussian(zyy_rec)
            app.logger.debug("Probability to be gaussian {}".format(pvalue))
            self.losses['kstestz'].append(pvalue)
            writer = self.writer_debug if trial_writer == None else trial_writer
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("common[z2]/zyx", zyx_rec[idx,:], epoch)
                if idx== 20: 
                    break
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("common[z2]/zxy", zxy_rec[idx,:], epoch)
                if idx== 20: 
                    break
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("specific[z1]/zyy", zyy_rec[idx,:], epoch)
                if idx== 20: 
                    break
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("specific[z1]/zxx", zxx_rec[idx,:], epoch)
                if idx== 20: 
                    break
        
        Gloss.backward()
        self.oGyx.step()
        self.gradients['Fxy'].append(self.track_gradient_change(self.Fxy.module))
        self.gradients['Gy'].append(self.track_gradient_change(self.Gy.module))
        zerograd(self.optz)
         
        self.losses['Gloss'      ].append(Gloss.tolist())
        self.losses['Gloss_ali'  ].append(Gloss_ali.tolist())
        self.losses['Gloss_ali_x'].append(Gloss_ali_x.tolist())
        self.losses['Gloss_ali_y'].append(Gloss_ali_y.tolist())

        self.losses['Gloss_marginal'    ].append(Gloss_marginal.tolist())
        self.losses['Gloss_marginal_y'  ].append(Gloss_marginal_y.tolist())
        self.losses['Gloss_marginal_zd' ].append(Gloss_marginal_zd.tolist())
        self.losses['Gloss_marginal_x'  ].append(Gloss_marginal_x.tolist())
        self.losses['Gloss_marginal_zf' ].append(Gloss_marginal_zf.tolist())

        self.losses['Gloss_rec'    ].append(Gloss_rec.tolist())
        self.losses['Gloss_rec_zd' ].append(Gloss_rec_zd.tolist())
        self.losses['Gloss_rec_y'  ].append(Gloss_rec_y.tolist())
        self.losses['Gloss_rec_zx' ].append(Gloss_rec_zx.tolist())
        self.losses['Gloss_rec_x'  ].append(Gloss_rec_x.tolist())
    
        self.losses['Gloss_rec_zxy'].append(Gloss_rec_zxy.tolist())

    def generate_latent_variable(self, batch, nch_zd,nzd, nch_zf = 128,nzf = 128):
        zyy  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        zxx  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        zyx  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        zxy  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        return zyy, zyx, zxx, zxy

    # @profile
    def train_unique(self):
        app.logger.info('Training on both recorded only signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)

        loader     =  self.trn_loader #if self.trial == None else self.tst_loader
        total_step = len(loader)

        app.logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        self.writer_histo = None

        if self.study_dir!=None:
            start_time          = time.ctime(time.time()).replace(' ','-').replace(':','_')
            writer_loss_dir     = f'{self.study_dir}/hparams/trial-{self.trial.number}/loss/evento-{start_time}/'
            writer_accuracy_dir = f'{self.study_dir}/hparams/trial-{self.trial.number}/accuracy/evento-{start_time}/'
            # writer_histo_dir    = f'{self.study_dir}/hparams/trial-{self.trial.number}/histogram/evento-{start_time}/'
            
            self.writer_loss     = SummaryWriter(writer_loss_dir)
            self.writer_accuracy = SummaryWriter(writer_accuracy_dir)
            self.writer_histo    = SummaryWriter(writer_accuracy_dir)

            app.logger.info(f'Tensorboard Writer setted up for trial {self.trial.number} ...')

        nch_zd, nzd = 4,128
        nch_zf, nzf = 4,128
        bar = trange(0,self.opt.niter)

        # if self.trial != None:
        #     #forcing the same seed to increase the research of good hyper parameter
        
        for epoch in bar:
            for b,batch in enumerate(loader):
                y,x, *others = batch
                y   = y.to(app.DEVICE, non_blocking = True)
                x   = x.to(app.DEVICE, non_blocking = True)
                
                zyy,zyx, *other = self.generate_latent_variable(
                            batch   = len(y),
                            nzd     = nzd,
                            nch_zd  = nch_zd,
                            nzf     = nzf,
                            nch_zf  = nch_zf)
                if torch.isnan(torch.max(y)):
                    app.logger.debug("your model contain nan value "
                        "this signals will be withdrawn from the training "
                        "but style be present in the dataset. \n"
                        "Then, remember to correct your dataset")
                    mask   = [not torch.isnan(torch.max(y[e,:])).tolist() for e in range(len(y))]
                    index  = np.array(range(len(y)))
                    y.data = y[index[mask]]
                    x.data = x[index[mask]]
                    zyy.data, zyx.data = zyy[index[mask]],zyx[index[mask]]
                
                for _ in range(1):
                    self.alice_train_discriminator_adv(y,zyy,zyx,x)                 
                for _ in range(2):
                    self.alice_train_generator_adv(y,zyy,zyx,x,epoch, self.writer_histo)
                app.logger.debug(f'Epoch [{epoch}/{opt.niter}]\tStep [{b}/{total_step-1}]')
                
                if epoch%20 == 0 and self.trial== None:
                    for k,v in self.losses.items():
                        self.writer_train.add_scalar('Loss/{}'.format(k),
                            np.mean(np.array(v[-b:-1])),epoch)

                    for k, v in self.gradients.items():
                        self.writer_debug.add_scalar('Gradient/{}'.format(k),
                            np.mean(np.array(v[-b:-1])),epoch)

            Gloss = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss'][-b:-1])))
            Dloss = '{:>5.3f}'.format(np.mean(np.array(self.losses['Dloss'][-b:-1])))
            Gloss_zxy = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss_rec_zxy'][-b:-1])))
            # self.writer_debug.add_scalars('Loss/Main',{'Dloss':Dloss,'Gloss':Gloss},epoch)
            # bar.set_postfix(Gloss = Gloss, Dloss = Dloss)

            bar.set_postfix(Gloss = Gloss, Gloss_zxy = Gloss_zxy, Dloss = Dloss) 
            if epoch%25 == 0 and self.trial == None:
                # for k,v in self.losses.items():
                #     self.writer_train.add_scalar('Loss/{}'.format(k),
                #         np.mean(np.array(v[-b:-1])),epoch)
                torch.manual_seed(100)
                figure_bb, gof_bb = plt.plot_generate_classic(
                        tag     = 'broadband',
                        Qec     = deepcopy(self.Fxy),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)

                bar.set_postfix(status = 'writing reconstructed broadband signals ...')
                self.writer_val.add_figure('Broadband',figure_bb, epoch)
                self.writer_val.add_figure('Goodness of Fit Broadband',gof_bb, epoch)
                figure_fl, gof_fl = plt.plot_generate_classic(
                        tag     = 'broadband',
                        Qec     = deepcopy(self.Fxy),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique_hack",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                bar.set_postfix(status = 'writing reconstructed filtered signals ...')
                self.writer_val.add_figure('Filtered',figure_fl, epoch)
                self.writer_val.add_figure('Goodness of Fit Filtered',gof_fl, epoch)

                figure_hf, gof_hf = plt.plot_generate_classic(
                        tag     = 'hybrid',
                        Qec     = deepcopy(self.Fxy),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                bar.set_postfix(status = 'writing reconstructed hybrid broadband signals ...')
                self.writer_val.add_figure('Hybrid (Filtered)',figure_hf, epoch)
                self.writer_val.add_figure('Goodness of Fit Hybrid (Filtered)',gof_hf, epoch)

                figure_hb, gof_hb = plt.plot_generate_classic(
                        tag     = 'hybrid',
                        Qec     = deepcopy(self.Fxy),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique_hack",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                bar.set_postfix(status = 'writing reconstructed hybrid broadband signals ...')
                self.writer_val.add_figure('Hybrid (Broadband)',figure_hb, epoch)
                self.writer_val.add_figure('Goodness of Fit Hybrid (Broadband)',gof_hb, epoch)
                
                # random.seed(opt.manualSeed)
            
            if epoch%20 == 0:
                val_accuracy_bb, val_accuracy_fl, val_accuracy_hb = self.accuracy()
                app.logger.info("val_accuracy broadband = {:>5.3f}".format(val_accuracy_bb))
                app.logger.info("val_accuracy filtered  = {:>5.3f}".format(val_accuracy_fl))
                app.logger.info("val_accuracy hybrid    = {:>5.3f}".format(val_accuracy_hb))
                bar.set_postfix(**{'val_accuracy_bb':val_accuracy_bb,
                                'val_accuracy_fl':val_accuracy_fl,
                                'val_accuracy_hb':val_accuracy_hb})
                
                if self.trial == None:
                    bar.set_postfix(status='tracking weight ...')
                    self.track_weight_change(
                                writer  = self.writer_debug,
                                tag     ='F[cnn_common]',
                                model   = self.Fxy.module.cnn_common.eval(), 
                                epoch   = epoch
                    )
                    self.track_weight_change(
                                writer  = self.writer_debug,
                                tag     ='F[cnn_broadband]',
                                model   = self.Fxy.module.cnn_broadband.eval(), 
                                epoch   = epoch
                    )
                    self.track_weight_change(
                                writer  = self.writer_debug,
                                tag     ='Gy',
                                model   = self.Gy.module.cnn1.eval(), 
                                epoch   = epoch
                    )
        
                if self.study_dir == None:
                    self.writer_debug.add_scalar('Accuracy/Broadband',val_accuracy_bb, epoch)
                    self.writer_debug.add_scalar('Accuracy/Filtered',val_accuracy_fl, epoch)
                    self.writer_debug.add_scalar('Accuracy/JHybrid',val_accuracy_hb, epoch)

                elif self.study_dir!=None:
                    self.writer_accuracy.add_scalar('Accuracy/Broadband',val_accuracy_bb,epoch)
                    self.writer_accuracy.add_scalar('Accuracy/Filtered',val_accuracy_fl,epoch)
                    self.writer_accuracy.add_scalar('Accuracy/Hybrid',val_accuracy_hb,epoch)
                    
                    self.writer_loss.add_scalar('Loss/Dloss',    float(Dloss),    epoch)
                    # self.writer_loss.add_scalar('Loss/Dloss_zxy',float(Dloss_zxy),epoch)
                    self.writer_loss.add_scalar('Loss/Gloss',    float(Gloss),    epoch)
                    self.writer_loss.add_scalar('Loss/Gloss_zxy',float(Gloss_zxy),epoch)
                    # bar.set_postfix(accuracy_fl = val_accuracy_fl)
                    # bar.set_postfix(accuracy_bb = val_accuracy_bb)
                    self.trial.report(val_accuracy_hb, epoch)
                    if self.trial.should_prune():
                        self.writer_hparams.add_hparams(
                        {
                            'rlr' : self.rlr, 
                            'glr' :self.glr,
                            'weight_decay':self.weight_decay
                        },
                        {
                            'hparams/broadband': val_accuracy_bb,
                            'hparams/filtered' : val_accuracy_fl,
                            'hparams/hybrid'   : val_accuracy_hb
                        })

                        raise optuna.exceptions.TrialPruned()
                else:
                    app.logger.info("No accuracy saved ...")
           
            if (epoch+1)%save_checkpoint == 0 and self.trial == None:
                app.logger.info(f"saving model at this checkpoint :{epoch}")
                
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Fxy.state_dict(),
                        'optimizer_state_dict'  : self.oGyx.state_dict(),
                        'loss'                  : self.losses,},
                        root_checkpoint+'/Fyx.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Gy.state_dict(),
                        'optimizer_state_dict'  : self.oGyx.state_dict(),
                        'loss'                  : self.losses,},
                        root_checkpoint +'/Gy.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dy.state_dict(),
                        'optimizer_state_dict'  : self.oDyxz.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dy.pth')
                # tsave({ 'epoch'                 : epoch,
                #         'model_state_dict'      : self.Dyy.state_dict(),
                #         'optimizer_state_dict'  : self.oDyxz.state_dict(),
                #         'loss'                  :self.losses,},
                #         root_checkpoint +'/Dyy.pth')
                # tsave({ 'epoch'                 : epoch,
                #         'model_state_dict'      : self.Dzzb.state_dict(),
                #         'optimizer_state_dict'  : self.oDyxz.state_dict(),
                #         'loss'                  :self.losses,},
                #         root_checkpoint +'/Dzzb.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dzb.state_dict(),
                        'optimizer_state_dict'  : self.oDyxz.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dzb.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dyz.state_dict(),
                        'optimizer_state_dict'  : self.oDyxz.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dyz.pth')
                # tsave({ 'epoch'                 : epoch,
                #         'model_state_dict'      : self.Dzyx.state_dict(),
                #         'optimizer_state_dict'  : self.oDyxz.state_dict(),
                #         'loss'                  :self.losses,},
                #         root_checkpoint +'/Dzyx.pth')

            if (epoch +1) == opt.niter:
                if self.trial == None:
                    for key, value in self.losses.items():
                        plt.plot_loss_explicit(losses=value, key=key, outf=outf,niter=niter)
                    app.logger.info("Training finishes !")
                    return self
                else:
                    self.writer_hparams.add_hparams(
                    {
                        'rlr'  : self.rlr, 
                        'glr'  : self.glr, 
                        'weight_decay':self.weight_decay
                    },
                    {
                        'hparams/broadband': val_accuracy_bb,
                        'hparams/filtered' : val_accuracy_fl,
                        'hparams/hybrid'   : val_accuracy_hb,
                    })
                    app.logger.info('Evaluating ...')
                    return val_accuracy_hb


    def track_weight_change(self, writer, tag, model,epoch):
        for idx in range(len(model)):
            classname = model[idx].__class__.__name__
            if classname.find('Conv1d')!= -1 or classname.find('ConvTranspose1d')!= -1:
                writer.add_histogram(f'{tag}/{idx}', model[idx].weight, epoch)
    
    def track_gradient_change(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
        
    def accuracy(self):
        def _eval(EG,PG): 
            val = np.sqrt(np.power([10 - eg for eg in EG],2)+\
                np.power([10 - pg for pg in PG],2))
            accuracy = val.mean().tolist()
            return accuracy

        EG_h, PG_h  = plt.get_gofs(tag = 'hybrid', 
            Qec = self.Fxy, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique",
            opt = opt,
            std = self.std, 
            outf = outf)

        EG_b, PG_b  = plt.get_gofs(tag = 'broadband', 
            Qec = self.Fxy, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique",
            opt = opt,
            std = self.std, 
            outf = outf)

        EG_f, PG_f  = plt.get_gofs(tag = 'broadband', 
            Qec = self.Fxy, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique_hack",
            opt = opt,
            std = self.std, 
            outf = outf)

        accuracy_hb = _eval(EG_h,PG_h)
        accuracy_bb = _eval(EG_b,PG_b)
        accuracy_fl = _eval(EG_f,PG_f)

        # if accuracy == np.nan:
        #     accuracy =10*np.sqrt(2)
        #     return accuracy
        return accuracy_bb, accuracy_fl,accuracy_hb



    # @profile
    def train(self):
        # breakpoint()
        for t,a in self.strategy['tract'].items():
            if 'unique' in t.lower() and a:
                self.train_unique()

    # @profile            
    def generate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        app.logger.info("generating result...")
        # breakpoint()
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'unique' in t and self.strategy['trplt']['unique']:
            plt.plot_generate_classic(tag = 'broadband',
                Qec     = self.Fxy,
                Pdc     = self.Gy,
                trn_set = self.vld_loader,
                pfx="vld_set_fl_unique",
                opt=opt,
                outf=outf)


