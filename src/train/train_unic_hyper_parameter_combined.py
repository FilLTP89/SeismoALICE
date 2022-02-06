# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from copy import deepcopy
from common.common_nn import *
from common.common_torch import * 
import plot.plot_tools as plt
import profiling.profile_support as profile
from tools.generate_noise import latent_resampling, noise_generator
from database.database_sae import random_split 
from database.database_sae import thsTensorData
import torch.nn as nn 
import json
import pdb
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
import time
import GPUtil
from database.toyset import Toyset, get_dataset
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
    '''Initialize neural network'''
    def __init__(self,cv, trial=None, study_dir=None):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        super(trainer, self).__init__()

        self.cv = cv
        self.gr_norm = []
        self.std_y = 1.0
        self.std_x = 1.0
        self.trial = trial
        self.study_dir = study_dir
        # rlr = 0.0003504249301844992
        # glr = 0.04850731189532911
        
        globals().update(cv)
        globals().update(opt.__dict__)

        if study_dir!=None:
            self.glr = self.trial.suggest_float("glrx",0.001, 0.1,log=True)
            self.rlr = self.trial.suggest_float("rlrx",0.001, 0.1,log=True)
        else:
            self.glr = opt.glr
            self.rlr = opt.rlr
            app.logger.info(f'glr = {self.glr}')
            app.logger.info(f'rlr = {self.rlr}')

        b1              = 0.5
        b2              = 0.9999
        self.strategy   = strategy
        self.opt        = opt
        self.start_epoch= 0  

        nzd = opt.nzd
        ndf = opt.ndf
        ngpu_use = torch.cuda.device_count()
        # torch.backends.cudnn.deterministic  = True
        # torch.backends.cudnn.benchmark      = False

        self.Dnets      = []
        self.Dnetsy     = []
        self.Dnetsx     = []
        self.optz       = []
        self.oGyx       = None
        _blocked        = False
        # self.dp_mode    = True
        self.losses     = {
            'Dloss':[0],

            'Dloss_y':[0],
            'Dloss_ali_y':[0],
            'Dloss_rec_y':[0],

            'Dloss_x':[0],
            'Dloss_ali_x':[0],
            'Dloss_rec_x':[0],
            
            'Dloss_identity_zxy':[0],
            
            'Gloss':[0],

            'Gloss_ali_y':[0],
            'Gloss_ali_x':[0],
            'Gloss_cycle_consistency_y':[0],
            'Gloss_cycle_consistency_zd':[0],
            'Gloss_identity_y':[0],
            'Gloss_identity_zd':[0],

            'Gloss_cycle_consistency_x':[0],
            'Gloss_cycle_consistency_zf':[0],
            'Gloss_identity_x':[0],
            'Gloss_identity_zf':[0],

            'Gloss_identity_zxy':[0],
        }

        network_path      = './network/unic/zyy24/nsy1280'
        
        if self.trial == None:
            summary_dir       = './runs_both/unic/tuning/'
            self.writer_train = SummaryWriter(f'{summary_dir}/training')
            self.writer_val   = SummaryWriter(f'{summary_dir}/validation')
            self.writer_debug = SummaryWriter(f'{summary_dir}/debug')
        else:
            hparams_dir         = f'{self.study_dir}/hparams/'
            self.writer_hparams = SummaryWriter(f'{hparams_dir}')
        # self.writer_debug_encoder = SummaryWriter('runs_both/filtered/tuning/debug/encoder')
        
        flagT=False
        flagF=False
        
        
        t = [y.lower() for y in list(self.strategy.keys())] 

        net = Network(DataParalleleFactory())

        self.trn_loader, self.vld_loader, self.tst_loader = trn_loader, vld_loader, tst_loader

        if 'unique' in t:
            self.style='ALICE'
            # act = acts[self.style]
            n = self.strategy['unique']

            self.F_  = net.Encoder(opt.config['F'],  opt)
            self.Gy  = net.Decoder(opt.config['Gy'], opt)
            self.Gx  = net.Decoder(opt.config['Gx'], opt)

            self.F_  = nn.DataParallel(self.F_).cuda()
            self.Gy  = nn.DataParallel(self.Gy).cuda()
            self.Gx  = nn.DataParallel(self.Gx).cuda()

            self.FGf  = [self.F_,self.Gy, self.Gx]

            if  self.strategy['tract']['unique']:
                if None in n:       
                    # self.FGf  = [self.F_, self.Gy]
                    
                    self.oGyx = reset_net(self.FGf,
                        func=set_weights,lr=self.glr,b1=b1,b2=b2,
                        weight_decay=0.00001)

                    # self.oGyx = Adam(ittc(self.F_.branch_common.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,betas=(b1,b2),
                    #     weight_decay=0.00001)

                    # self.oGx = Adam(ittc(self.Gx.parameters()),
                    #     lr=glrx,betas=(b1,b2),
                    #     weight_decay=0.00001)
                else: 
                    # breakpoint()
                    self.F_.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    # self.Gx.load_state_dict(tload(n[2])['model_state_dict'])
                    if not _blocked:
                        app.logger.info("considering master part as pretrained : no grad required")
                        for param in self.F_.module.master.parameters():
                            param.requires_grad = False

                        app.logger.info("considering common part as pretrained : no grad required")
                        for param in self.F_.module.cnn_common.parameters():
                            param.requires_grad = False

                        self.oGyx = Adam(ittc(self.F_.parameters()),
                            lr=glr,betas=(b1,b2),
                            weight_decay=0.00001)
                        # self.oGy = Adam(ittc(self.F_.branch_broadband.parameters(),
                        #     self.Gy.parameters()),
                        #     lr=ngpu_use*glr,betas=(b1,b2),
                            # weight_decay=0.00001)
                        
                        self.FGf  = [self.F_,self.Gy]
                        app.logger.info("intialization of Gy and the broadband branch for the broadband training ...")
                        self.oGyx = reset_net([self.Gy,self.F_.module.cnn_broadband],
                            func=set_weights,lr=self.glr,b1=b1,b2=b2,
                            weight_decay=0.00001)
                    else: 
                        app.logger.info("starting for a previous trained auto-encoder without block back propagation")
                        self.oGy = Adam(ittc(self.F_.parameters(),self.Gy.parameters()),
                            lr=self.glr,betas=(b1,b2),
                            weight_decay=0.00001)

                        self.oGx = reset_net([self.Gx],
                        func=set_weights,lr=self.glr,b1=b1,b2=b2,
                        weight_decay=0.00001)

                        self.FGf  = [self.F_,self.Gy]
                        self.FGx  = [self.Gx]
                    # self.oGyx = RMSProp(ittc(self.F_.parameters(),
                    #     self.Gy.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,alpha=b2,
                    #     weight_decay=0.00001)
                self.optz.append(self.oGyx)
                # self.optz.append(self.oGx)

                self.Dy   = net.DCGAN_Dx( opt.config['Dy'],  opt)
                self.Dzb  = net.DCGAN_Dz( opt.config['Dzb'], opt)
                self.Dyz  = net.DCGAN_DXZ(opt.config['Dyz'], opt)
                self.Dzzb = net.DCGAN_Dz( opt.config['Dzzb'],opt)
                self.Dyy  = net.DCGAN_Dx( opt.config['Dyy'], opt)

                self.Dzyx = net.DCGAN_Dz(opt.config['Dzyx'],opt)

                self.Dx   = net.DCGAN_Dx(opt.config['Dx'],  opt)
                self.Dzf  = net.DCGAN_Dz(opt.config['Dzf'], opt)
                self.Dxz  = net.DCGAN_DXZ(opt.config['Dxz'],opt)
                self.Dzzf = net.DCGAN_Dz(opt.config['Dzzf'],opt)
                self.Dxx  = net.DCGAN_Dx(opt.config['Dxx'], opt)

                self.Dy   = nn.DataParallel(self.Dy  ).cuda()
                self.Dzb  = nn.DataParallel(self.Dzb ).cuda()
                self.Dyz  = nn.DataParallel(self.Dyz ).cuda()
                self.Dzzb = nn.DataParallel(self.Dzzb).cuda()
                self.Dyy  = nn.DataParallel(self.Dyy ).cuda()

                if _blocked:
                    self.Dy.load_state_dict(tload(f'{network_path}/Dy.pth')['model_state_dict'])
                    self.Dzb.load_state_dict(tload(f'{network_path}/Dzb.pth')['model_state_dict'])
                    self.Dyz.load_state_dict(tload(f'{network_path}/Dyz.pth')['model_state_dict'])
                    self.Dzzb.load_state_dict(tload(f'{network_path}/Dzzb.pth')['model_state_dict'])
                    self.Dyy.load_state_dict(tload(f'{network_path}/Dyy.pth')['model_state_dict'])
                
                self.Dzyx = nn.DataParallel(self.Dzyx ).cuda()

                self.Dx   = nn.DataParallel(self.Dx   ).cuda()
                self.Dzf  = nn.DataParallel(self.Dzf  ).cuda()
                self.Dxz  = nn.DataParallel(self.Dxz  ).cuda()
                self.Dzzf = nn.DataParallel(self.Dzzf ).cuda()
                self.Dxx  = nn.DataParallel(self.Dxx  ).cuda()

                self.Dnetsy.append(self.Dy   )
                self.Dnetsy.append(self.Dzb  )
                self.Dnetsy.append(self.Dyz  )
                self.Dnetsy.append(self.Dzzb )
                self.Dnetsy.append(self.Dyy  )
                
                self.Dnetsx.append(self.Dzyx )

                self.Dnetsx.append(self.Dx  )
                self.Dnetsx.append(self.Dzf )
                self.Dnetsx.append(self.Dxz )
                self.Dnetsx.append(self.Dzzf)
                self.Dnetsx.append(self.Dxx )

                if _blocked:
                    self.oDy = Adam(ittc(self.Dy.parameters(),
                                    self.Dzb.parameters(),
                                    self.Dyz.parameters(),
                                    self.Dzzb.parameters(),
                                    self.Dyy.parameters()),
                                lr=self.rlr,betas=(b1,b2),
                                weight_decay=0.00001)
                else:
                    self.oDy = reset_net(self.Dnetsy,
                    func=set_weights,lr=self.rlr,
                    optim='Adam', b1 = b1, b2 = b2,
                    weight_decay=0.00001)

                self.oDx = reset_net(self.Dnetsx,
                    func=set_weights,lr=self.rlr,
                    optim='Adam', b1 = b1, b2 = b2,
                    weight_decay=0.00001)

                # self.oDx  = reset_net(self.Dnetsx,
                #     func=set_weights,lr = rlrx,
                #     optim='Adam', b1 = b1, b2 = b2,
                #     weight_decay=0.00001)
                # self.oDyxz = self.oDy + self.oDx
                self.optz.append(self.oDy)
                self.optz.append(self.oDx)

            else:
                if None not in n:
                    checkpoint          = tload(n[0])
                    self.start_epoch    = checkpoint['epoch']
                    self.losses         = checkpoint['loss']
                    self.F_.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    self.FGf  = [self.F_,self.Gy]
                    # self.Gx.load_state_dict(tload(n[2])['model_state_dict'])  
                else:
                    flagF=False

        
        # self.writer_debug_encoder.add_graph(next(iter(self.F_.children())),torch.randn(128,6,4096).cuda())
        # self.writer_debug_decoder.add_graph(next(iter(self.Gy.children())), torch.randn(128,512,256).cuda())
        self.bce_loss = BCE(reduction='mean')
        self.l2_loss  = MSELoss(reduction='mean')
        # breakpoint()
        print("Parameters of Encoder/Decoders ")
        count_parameters(self.FGf)
        # count_parameters(self.FGx)
        print("Parameters of Discriminators ")
        count_parameters(self.Dnetsy)
        count_parameters(self.Dnetsx)
        
        
       
    def discriminate_xz(self,x,xr,z,zr):
        # Discriminate real
        ftz = self.Dzf(zr) #OK: no batchNorm
        ftx = self.Dx(x) #with batchNorm
        zrc = zcat(ftx,ftz)
        ftxz = self.Dxz(zrc) #no batchNorm
        Dxz = ftxz

        # Discriminate fake
        ftz = self.Dzf(z)
        ftx = self.Dx(xr)
        zrc = zcat(ftx,ftz)
        ftzx = self.Dxz(zrc)
        Dzx  = ftzx

        return Dxz,Dzx #,ftr,ftf

    def discriminate_yz(self,y,yr,z,zr):
        # Discriminate real
        
        ftz = self.Dzb(zr) #OK : no batchNorm
        ftx = self.Dy(y) #OK : with batchNorm
        zrc = zcat(ftx,ftz)
        ftxz = self.Dyz(zrc)
        Dxz  = ftxz
        
        # Discriminate fake
        ftz = self.Dzb(z)
        ftx = self.Dy(yr)
        zrc = zcat(ftx,ftz)
        ftzx = self.Dyz(zrc)
        Dzx  = ftzx

        return Dxz,Dzx 

    def discriminate_xx(self,x,xr):
        # x and xr should have the same distribution !
        Dreal = self.Dxx(zcat(x,x))#with batchNorm
        Dfake = self.Dxx(zcat(x,xr))
        return Dreal,Dfake

    def discriminate_yy(self,y,yr):
        Dreal = self.Dyy(zcat(y,y )) #with batchNorm
        Dfake = self.Dyy(zcat(y,yr))
        return Dreal,Dfake

    def discriminate_zzb(self,z,zr):
        Dreal = self.Dzzb(zcat(z,z )) #no batchNorm
        Dfake = self.Dzzb(zcat(z,zr))
        return Dreal,Dfake

    def discriminate_zzf(self,z,zr):
        Dreal = self.Dzzf(zcat(z,z )) #no batchNorm
        Dfake = self.Dzzf(zcat(z,zr))
        return Dreal,Dfake

    def discriminate_zxy(self,z_yx,z_xy):
        D_zyx = self.Dzyx(z_yx)
        D_zxy = self.Dzyx(z_xy)
        return D_zyx,D_zxy
    
    # @profile
    def alice_train_discriminator_adv(self,y,zyy,zyx, x, zxy):
         # Set-up training
        # breakpoint()
        zerograd(self.optz)
        modalite([self.F_],   mode = 'eval')
        modalite([self.Gy],   mode = 'eval')
        modalite([self.Gx],   mode = 'eval')
        modalite(self.Dnetsy, mode = 'train')
        modalite(self.Dnetsx, mode = 'train')
        
        # I. Broadband training of Discriminator

        # 0. Generate noise
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std_y})

        # 1. Concatenate inputs
        z_inp = zcat(zyx,zyy)
        y_inp = zcat(y,wny)
        

        # 2.1 Generate conditional samples
        y_gen   = self.Gy(z_inp)
        app.logger.debug(f'max y_inp : {torch.max(y_inp)}')
        zyy_F,zyx_F,*other = self.F_(y_inp)
        #2.2 Concatanete outputs
        app.logger.debug(f'max zyx_F : {torch.max(zyx_F)}')
        app.logger.debug(f'max zyy_F : {torch.max(zyy_F)}')
        zyy_gen = zcat(zyx_F,zyy_F)
        zyx_gen = zcat(zyx_F)
        # 3. Cross-Discriminate YZ
        Dyz,Dzy = self.discriminate_yz(y,y_gen,z_inp,zyy_gen)

        # 4. Compute ALI discriminator loss
        Dloss_ali_y = -torch.mean(ln0c(Dzy)+ln0c(1.0-Dyz))
        
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std_y})

        # 5. Generate reconstructions
        y_rec  = self.Gy(zyy_gen)
        app.logger.debug(f"max zyy_gen : {torch.max(zyy_gen)}")
        app.logger.debug(f'max y       : {torch.max(y_rec)}')
        assert torch.isfinite(torch.max(y_rec))

        y_gen  = zcat(y_gen,wny)
        zyy_F,zyx_F,*other = self.F_(y_gen)
        zyy_rec = zcat(zyx_F,zyy_F)

        # 6. Disciminate Cross Entropy  
        Dreal_y,Dfake_y     = self.discriminate_yy(y,y_rec)
        Dloss_rec_y         = self.bce_loss(Dreal_y,o1l(Dreal_y))+\
                              self.bce_loss(Dfake_y,o0l(Dfake_y))
        # Dloss_rec_y         = -torch.mean(ln0c(Dreal_y)+ln0c(1.0-Dfake_y))

        Dreal_zd,Dfake_zd   = self.discriminate_zzb(z_inp,zyy_rec)
        Dloss_rec_zy        = self.bce_loss(Dreal_zd,o1l(Dreal_zd))+\
                              self.bce_loss(Dfake_zd,o0l(Dfake_zd))
        # Dloss_rec_zy        = -torch.mean(ln0c(Dreal_zd)+ln0c(1.0-Dfake_zd))
        
        #7. Compute all losses
        Dloss_rec_bb        = Dloss_rec_y + Dloss_rec_zy
        Dloss_ali_bb        = Dloss_ali_y 
        Dloss_y             = Dloss_ali_bb + Dloss_rec_bb


        ## II. filtered part of training

        wnx,wnz,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std_x})

        # 1. Concatenate inputs
        zf_inp = zcat(zxy,wnz)
        x_inp  = zcat(x,wnx)
        
        # 2.1 Generate conditional samples
        x_gen = self.Gx(zf_inp)
        # app.logger.debug(f'max y_inp : {torch.max(x_inp)}')
        _,zyx_F,*other = self.F_(x_inp)
        #2.2 Concatanete outputs
        app.logger.debug(f'max zyx_F : {torch.max(zyx_F)}')
        _zxy_gen = zcat(zyx_F)

        # 3. Cross-Discriminate YZ
        Dxz,Dzx = self.discriminate_xz(x,x_gen,zxy,_zxy_gen)

        # 4. Compute ALI discriminator loss
        Dloss_ali_x = -torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        
        wnx,wnz,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std_x})

        # 5. Generate reconstructions
        zxy_gen = zcat(_zxy_gen,wnz)
        x_rec   = self.Gx(zxy_gen)
        x_gen   = zcat(x_gen,wnx)
        _,zyx_F,*other = self.F_(x_gen)
        zxy_rec = zcat(zyx_F)

        # 6. Disciminate Cross Entropy  
        Dreal_x,Dfake_x     = self.discriminate_xx(x,x_rec)
        Dloss_rec_x         = self.bce_loss(Dreal_x,o1l(Dreal_x))+\
                              self.bce_loss(Dfake_x,o0l(Dfake_x))
        # Dloss_rec_y         = -torch.mean(ln0c(Dreal_y)+ln0c(1.0-Dfake_y))

        Dreal_zf,Dfake_zf   = self.discriminate_zzf(zxy,zxy_rec)
        Dloss_rec_zx        = self.bce_loss(Dreal_zf,o1l(Dreal_zf))+\
                              self.bce_loss(Dfake_zf,o0l(Dfake_zf))
        # Dloss_rec_zy        = -torch.mean(ln0c(Dreal_zd)+ln0c(1.0-Dfake_zd))
        
        # 7. Compute all losses
        Dloss_rec_fl         = Dloss_rec_x + Dloss_rec_zx
        Dloss_ali_fl         = Dloss_ali_x 
        Dloss_fl             = Dloss_ali_fl + Dloss_rec_fl
        

        ## II. Total loss 
        Dloss_x               = Dloss_fl

        # III. Common latent space
        # zyx_gen = zyx_gen
        # zxy_gen = zxy_gen
        Dzyx, Dzxy          = self.discriminate_zxy(zyx_gen,_zxy_gen)
        Dloss_identity_zxy  = self.bce_loss(Dzxy,o1l(Dzxy))+\
                              self.bce_loss(Dzyx,o0l(Dzyx))
        
        
        ## IV. Total loss 
        Dloss               = Dloss_x + Dloss_y + Dloss_identity_zxy

        Dloss.backward()
        self.oDy.step(),
        self.oDx.step(),
        # clipweights(self.Dnets), 
        clipweights(self.Dnetsx), 
        clipweights(self.Dnetsy), 
        zerograd(self.optz)

        self.losses['Dloss'].append(Dloss.tolist())

        # self.losses['Dloss_y'       ].append(Dloss_y.tolist())
        # self.losses['Dloss_ali_y'   ].append(Dloss_ali_y.tolist()) 
        # self.losses['Dloss_rec_y'   ].append(Dloss_rec_y.tolist())

        # self.losses['Dloss_x'       ].append(Dloss_x.tolist())
        # self.losses['Dloss_ali_x'   ].append(Dloss_ali_x.tolist()) 
        # self.losses['Dloss_rec_x'   ].append(Dloss_rec_x.tolist())

        self.losses['Dloss_identity_zxy'].append(Dloss_identity_zxy.tolist())

    # @profile
    def alice_train_generator_adv(self,y,zyy, zyx, x,zxy, epoch = None,trial_writer=None,*args,**kwargs):
        # Set-up training
        zerograd(self.optz)
        modalite([self.F_],   mode ='train')
        modalite([self.Gy],   mode ='train')
        modalite([self.Gx],   mode ='train')
        modalite(self.Dnetsy, mode ='train')
        modalite(self.Dnetsx, mode ='train')

        ## I. Broadband part
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std_y})
        
        # 1. Concatenate inputs
        y_inp  = zcat(y,wny)
        z_inp  = zcat(zxy,zyy)
         
        # 2. Generate conditional samples
        y_gen = self.Gy(z_inp)
        zyy_F,zyx_F,*other = self.F_(y_inp) 
 
        zyy_gen = zcat(zyx_F,zyy_F) 
        zyx_gen = zcat(zyx_F)
        Dyz,Dzy = self.discriminate_yz(y,y_gen,z_inp,zyy_gen)
        
        Gloss_ali_y =  torch.mean(-Dyz+Dzy) 
        
        # 3. Generate noise
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std_y})

        # 4. Generate reconstructions
        y_rec = self.Gy(zyy_gen)
        
        y_gen = zcat(y_gen,wny)
        zyy_F,zyx_F,*other = self.F_(y_gen)
        zyy_rec = zcat(zyx_F,zyy_F)
    
        # 5. Cross-Discriminate XX
        _,Dfake_y = self.discriminate_yy(y,y_rec)
        Gloss_cycle_consistency_y   = self.bce_loss(Dfake_y,o1l(Dfake_y))
        Gloss_identity_y            = torch.mean(torch.abs(y-y_rec)) 
        
        # 6. Cross-Discriminate ZZ
        _,Dfake_zd = self.discriminate_zzb(z_inp,zyy_rec)
        Gloss_cycle_consistency_zd  = self.bce_loss(Dfake_zd,o1l(Dfake_zd))
        Gloss_identity_zd           = torch.mean(torch.abs(z_inp-zyy_rec))

        # 7. Total Loss
        Gloss_cycle_consistency_y   = Gloss_cycle_consistency_y +  Gloss_cycle_consistency_zd
        Gloss_identity_y            = Gloss_identity_y + Gloss_identity_zd
        Gloss_y                     = (
                                        Gloss_ali_y + 
                                        Gloss_cycle_consistency_y*app.LAMBDA_CONSISTENCY + 
                                        Gloss_identity_y*app.LAMBDA_IDENTITY)

        # II. Filtered part
        wnx,wnz,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std_x})
        
        # 1. Concatenate inputs
        x_inp   = zcat(x,wnx)
        zf_inp  = zcat(zxy,wnz)
         
        # 2. Generate conditional samples
        x_gen = self.Gx(zf_inp)
        _,zyx_F,*other = self.F_(x_inp) 
        _zxy_gen = zcat(zyx_F) 
        Dxz,Dzx = self.discriminate_xz(x,x_gen,zxy,_zxy_gen)
        
        Gloss_ali_fl =  torch.mean(-Dxz+Dzx) 
        
        # 3. Generate noise
        wnx,wnz,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std_x})

        # 4. Generate reconstructions*
        zxy_gen = zcat(_zxy_gen,wnz)
        x_rec   = self.Gx(zxy_gen)
        x_gen   = zcat(x_gen,wnx)
        _,zyx_F,*other = self.F_(x_gen)
        zxy_rec = zcat(zyx_F)
    
        # 5. Cross-Discriminate XX
        _,Dfake_x = self.discriminate_xx(x,x_rec)
        Gloss_cycle_consistency_x   = self.bce_loss(Dfake_x,o1l(Dfake_x))
        Gloss_identity_x            = torch.mean(torch.abs(x-x_rec)) 
        
        # 6. Cross-Discriminate ZZ
        _,Dfake_zf = self.discriminate_zzf(zxy,zxy_rec)
        Gloss_cycle_consistency_zf  = self.bce_loss(Dfake_zf,o1l(Dfake_zf))
        Gloss_identity_zf           = torch.mean(torch.abs(zxy-zxy_rec)**2)

        # 7. Total Loss
        Gloss_cycle_consistency_fl   = Gloss_cycle_consistency_x +  Gloss_cycle_consistency_zf
        Gloss_identity_fl            = Gloss_identity_x + Gloss_identity_zf
        Gloss_x                      = (
                                        Gloss_ali_fl + 
                                        Gloss_cycle_consistency_fl*app.LAMBDA_CONSISTENCY + 
                                        Gloss_identity_fl*app.LAMBDA_IDENTITY
                                    )

        # III. Common latent space
        Dzyx, Dzxy                  = self.discriminate_zxy(zyx_gen, _zxy_gen)
        Gloss_identity_zxy          = torch.mean(torch.abs(Dzxy - Dzyx))
        Gloss_identity_zxy          = Gloss_identity_zxy*app.GAMMA
        # Gloss_identity_zxy          = 10.*self.bce_loss(Dzxy,o1l(Dzxy))+\
        #                               self.bce_loss(Dzyx,o0l(Dzyx))
        
        ## IV. Total loss
        # breakpoint()
        Gloss                       = Gloss_x + Gloss_y + Gloss_identity_zxy

        Gloss.backward()
        self.oGyx.step()
        zerograd(self.optz)

        if epoch%50 == 0: 
            writer = self.writer_debug if trial_writer == None else trial_writer
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("common/zxy", zxy_gen[idx,:], epoch)
                writer.add_histogram("common/zyx", zyx_gen[idx,:], epoch)

        self.losses['Gloss'].append(Gloss.tolist())

        # self.losses['Gloss_ali_y'].append(Gloss_ali_y.tolist())
        # self.losses['Gloss_ali_x'].append(Gloss_ali_x.tolist())

        # self.losses['Gloss_cycle_consistency_y' ].append(Gloss_cycle_consistency_y.tolist())
        # self.losses['Gloss_cycle_consistency_zd'].append(Gloss_cycle_consistency_zd.tolist())
        # self.losses['Gloss_identity_y'          ].append(Gloss_identity_y.tolist())
        # self.losses['Gloss_identity_zd'         ].append(Gloss_identity_zd.tolist())

        # self.losses['Gloss_cycle_consistency_x' ].append(Gloss_cycle_consistency_x.tolist())
        # self.losses['Gloss_cycle_consistency_zf'].append(Gloss_cycle_consistency_zf.tolist())
        # self.losses['Gloss_identity_x'          ].append(Gloss_identity_x.tolist())
        # self.losses['Gloss_identity_zf'         ].append(Gloss_identity_zf.tolist())
        
        self.losses['Gloss_identity_zxy'        ].append(Gloss_identity_zxy.tolist())

    def generate_latent_variable(self, batch, nch_zd,nzd, nch_zf = 128,nzf = 128):
        zyy  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=self.std_y).to(app.DEVICE, non_blocking = True)
        zxx  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=self.std_x).to(app.DEVICE, non_blocking = True)

        zyx  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=self.std_x).to(app.DEVICE, non_blocking = True)
        zxy  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=self.std_x).to(app.DEVICE, non_blocking = True)
        return zyy, zyx, zxy, zxx

    # @profile
    def train_unique(self):
        app.logger.info('Training on both recorded only signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)

        total_step = len(self.trn_loader)
        app.logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        if self.study_dir!=None:
            start_time          = time.ctime(time.time()).replace(' ','-').replace(':','_')
            writer_loss_dir     = f'{self.study_dir}/hparams/trial-{self.trial.number}/loss/evento-{start_time}/'
            writer_accuracy_dir = f'{self.study_dir}/hparams/trial-{self.trial.number}/accuracy/evento-{start_time}/'
            writer_histo_dir    = f'{self.study_dir}/hparams/trial-{self.trial.number}/histogram/evento-{start_time}/'
            
            self.writer_loss     = SummaryWriter(writer_loss_dir)
            self.writer_accuracy = SummaryWriter(writer_accuracy_dir)
            self.writer_histo    = SummaryWriter(writer_accuracy_dir)

            app.logger.info(f'Tensorboard accuracy setted up for trial {self.trial.number} ...')

        # bar = trange(0,10)
        nch_zd, nzd = 16,128
        nch_zf, nzf =  8,128
        bar = trange(opt.niter)
        # torch.cuda.empty_cache()
        torch.random.manual_seed(0)
        for epoch in bar:
            # breakpoint()
            for b,batch in enumerate(self.trn_loader):
                y,x, *others = batch
                # y   = y.to(app.DEVICE, non_blocking = True)
                y   = y.to(app.DEVICE, non_blocking = True)
                x   = x.to(app.DEVICE, non_blocking = True)
                # getting noise shape
                zyy,zyx,zxy, *other = self.generate_latent_variable(
                            batch   = len(y),
                            nzd     = nzd,
                            nch_zd  = nch_zd,
                            nzf     = nzf,
                            nch_zf  = nch_zf)
                # breakpoint()
                if torch.isnan(torch.max(y)):
                    app.logger.debug("your model contain nan value "
                        "this signals will be withdrawn from the training "
                        "but style be present in the dataset. \n"
                        "Then, remember to correct your dataset")
                    mask   = [not torch.isnan(torch.max(y[e,:])).tolist() for e in range(len(y))]
                    index  = np.array(range(len(y)))
                    y.data = y[index[mask]]
                    x.data = x[index[mask]]
                    zyy.data, zyx.data, zxy.data = zyy[index[mask]],zyx[index[mask]], zxy[index[mask]] 
    
                
                for _ in range(5):
                    self.alice_train_discriminator_adv(y,zyy,zyx,x,zxy)                 
                for _ in range(1):
                    self.alice_train_generator_adv(y,zyy,zyx,x,zxy,epoch, self.writer_histo)
                app.logger.debug(f'Epoch [{epoch}/{opt.niter}]\tStep [{b}/{total_step-1}]')
            # if epoch%10== 0:
            #     torch.manual_seed(100)
                if self.trial == None:
                    for k,v in self.losses.items():
                        self.writer_train.add_scalar('Loss/{}'.format(k),
                            np.mean(np.array(v[-b:-1])),epoch)
             
            Gloss       = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss'][-b:-1])))
            Gloss_zxy   = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss_identity_zxy'][-b:-1])))
            Dloss       = '{:>5.3f}'.format(np.mean(np.array(self.losses['Dloss'][-b:-1])))
            Dloss_zxy   = '{:>5.3f}'.format(np.mean(np.array(self.losses['Dloss_identity_zxy'][-b:-1])))
            
            bar.set_postfix(Gloss = Gloss, Gloss_zxy = Gloss_zxy, Dloss = Dloss, Dloss_zxy = Dloss_zxy) 
            
            if epoch%100 == 0 and self.trial == None:
                for k,v in self.losses.items():
                    self.writer_train.add_scalar('Loss/{}'.format(k),
                        np.mean(np.array(v[-b:-1])),epoch)
                torch.manual_seed(100)
                figure_bb, gof_bb = plt.plot_generate_classic(
                        tag     = 'broadband',
                        Qec     = deepcopy(self.F_),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                bar.set_postfix(status = 'writing reconstructed broadband signals ...')
                self.writer_val.add_figure('Broadband',figure_bb,epoch)
                self.writer_val.add_figure('Goodness of Fit Broadband',gof_bb,epoch)

                figure_fl, gof_fl = plt.plot_generate_classic(
                        tag     = 'filtered',
                        Qec     = deepcopy(self.F_),
                        Pdc     = deepcopy(self.Gx),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_fl_unique",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                bar.set_postfix(status = 'Writing reconstructed filtered signals ...')
                self.writer_val.add_figure('Filtered',figure_fl,epoch)
                self.writer_val.add_figure('Goodness of Fit Filtered',gof_fl,epoch)

            if (epoch+1)%25 == 0:
                val_accuracy, val_accuracy_bb, val_accuracy_fl = self.accuracy()
                # val_accuracy = self.accuracy()
                # app.logger.info(f"val_accuracy_fl = {val_accuracy}")
                app.logger.info(f"val_accuracy_fl = {val_accuracy_fl}")
                app.logger.info(f"val_accuracy_bb = {val_accuracy_bb}")
                if self.study_dir == None:
                    self.writer_debug.add_scalar('Accuracy/Broadband',val_accuracy_bb, epoch)
                    self.writer_debug.add_scalar('Accuracy/Filtered' ,val_accuracy_fl, epoch)
                    self.writer_debug.add_scalar('Accuracy/Total'    ,val_accuracy   , epoch)
                elif self.study_dir!=None:
                    self.writer_accuracy.add_scalar('Accuracy/Filtered' ,val_accuracy_fl,epoch)
                    self.writer_accuracy.add_scalar('Accuracy/Broadband',val_accuracy_bb,epoch)
                    self.writer_accuracy.add_scalar('Accuracy/Total'    ,val_accuracy   ,epoch)
                    
                    self.writer_loss.add_scalar('Loss/Dloss',    float(Dloss),    epoch)
                    self.writer_loss.add_scalar('Loss/Dloss_zxy',float(Dloss_zxy),epoch)
                    self.writer_loss.add_scalar('Loss/Gloss',    float(Gloss),    epoch)
                    self.writer_loss.add_scalar('Loss/Gloss_zxy',float(Gloss_zxy),epoch)
                    # bar.set_postfix(accuracy_fl = val_accuracy_fl)
                    # bar.set_postfix(accuracy_bb = val_accuracy_bb)
                    self.trial.report(val_accuracy, epoch)
                    if self.trial.should_prune():
                        self.writer_hparams.add_hparams(
                        {'rlr' : self.rlr, 'glr':self.glr},
                        {'hparams/total'    : val_accuracy, 
                         'hparams/filtered' : val_accuracy_fl,
                         'hparams/broadband': val_accuracy_bb})
                        raise optuna.exceptions.TrialPruned()
                else:
                    app.logger.info("No accuracy saved")
           
            if epoch%save_checkpoint == 0 and self.trial == None:
                app.logger.info(f"Saving model at this checkpoint :{epoch}")
                
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.F_.state_dict(),
                        'optimizer_state_dict'  : self.oGx.state_dict(),
                        'loss'                  : self.losses,},
                        root_checkpoint+'/Fyx.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Gx.state_dict(),
                        'optimizer_state_dict'  : self.oGx.state_dict(),
                        'loss'                  : self.losses,},
                        root_checkpoint +'/Gx.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dx.state_dict(),
                        'optimizer_state_dict'  : self.oDx.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dx.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dxx.state_dict(),
                        'optimizer_state_dict'  : self.oDx.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dxx.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dzzf.state_dict(),
                        'optimizer_state_dict'  : self.oDx.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dzzf.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dzf.state_dict(),
                        'optimizer_state_dict'  : self.oDx.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dzf.pth')
                # tsave({ 'epoch'                 : epoch,
                #         'model_state_dict'      : self.Dyz.state_dict(),
                #         'optimizer_state_dict'  : self.oDyxz.state_dict(),
                #         'loss'                  :self.losses,},
                #         root_checkpoint +'/Dyz.pth')
        if (epoch +1) == opt.niter:
            if self.trial == None:
                for key, value in self.losses.items():
                    plt.plot_loss_explicit(losses=value, key=key, outf=outf,niter=niter)
            elif self.trial!= None:
                self.writer_hparams.add_hparams(
                        {'rlr'  : self.rlr, 
                         'glr'  : self.glr},
                        {'hparams/total'    : val_accuracy, 
                         'hparams/filtered' : val_accuracy_fl,
                         'hparams/broadband': val_accuracy_bb})
                    
                app.logger.info('Evaluating ...')
                return val_accuracy
            else:
                app.logger.info("Training finishes !")
        
        

    def accuracy(self):
        EG_b, PG_b  = plt.get_gofs(tag = 'broadband', 
            Qec = self.F_, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique",
            opt = opt,
            std = self.std_y, 
            outf = outf)

        EG_f, PG_f  = plt.get_gofs(tag = 'filtered', 
            Qec = self.F_, 
            Pdc = self.Gx , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique",
            opt = opt,
            std = self.std_x, 
            outf = outf)

        val_b       = np.sqrt(np.power([10 - eg for eg in EG_b],2)+\
                     np.power([10 - pg for pg in PG_b],2))
        accuracy_bb = val_b.mean().tolist()

        val_f       = np.sqrt(np.power([10 - eg for eg in EG_f],2)+\
                     np.power([10 - pg for pg in PG_f],2))
        accuracy_fl = val_f.mean().tolist()

        accuracy    = max(accuracy_fl,accuracy_bb)
        # return accuracy_fl
        return accuracy, accuracy_bb, accuracy_fl



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
                Qec     = self.F_,
                Pdc     = self.Gy,
                trn_set = self.vld_loader,
                pfx="vld_set_fl_unique",
                opt=opt,
                outf=outf)
            
            plt.plot_generate_classic(tag = 'filtered',
                Qec     = self.F_, 
                Pdc     =self.Gx,
                trn_set = self.vld_loader,
                pfx="vld_set_fl_unique",
                opt=opt, outf=outf)

        if 'hybrid' in t and self.strategy['trplt']['hybrid']:
            n = self.strategy['hybrid']
            Fef = deepcopy(self.Fef)
            Gy = deepcopy(self.Gy)
            Ghz = deepcopy(self.Ghz)
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                Fef.load_state_dict(tload(n[0])['model_state_dict'])
                Gy.load_state_dict(tload(n[1])['model_state_dict'])
                Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            plt.plot_generate_hybrid(Fef,Gy,Ghz,app.DEVICE,vtm,\
                                      trn_loader,pfx="trn_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gy,Ghz,app.DEVICE,vtm,\
                                      tst_loader,pfx="tst_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gy,Ghz,app.DEVICE,vtm,\
                                      vld_loader,pfx="vld_set_hb",outf=outf)

    # @profile            
    def compare(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trcmp']['hybrid']:
            n = self.strategy['hybrid']
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            plt.plot_compare_ann2bb(self.Fef,self.Gy,self.Ghz,app.DEVICE,vtm,\
                                    trn_loader,pfx="trn_set_ann2bb",outf=outf)
    # @profile            
    def discriminate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trdis']['hybrid']:
            n = self.strategy['hybrid']
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
                self.Ddxz.load_state_dict(tload(n[3])['model_state_dict'])
                self.Dy.load_state_dict(tload(n[4])['model_state_dict'])
                self.Dzd.load_state_dict(tload(n[5])['model_state_dict'])
                # import pdb
                #pdb.set_trace()
                DsXz = load_state_dict(tload(n[6])['model_state_dict'])
            # Set-up training
            self.Fef.eval(),self.Gy.eval()
            self.Dy.eval(),self.Dzd.eval(),self.Ddxz.eval()
            
            for epoch in range(niter):
                for b,batch in enumerate(trn_loader):
                    # Load batch
                    xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
                    Xd = Variable(xd_data).to(app.DEVICE) # BB-signal
                    Xf = Variable(xf_data).to(app.DEVICE) # LF-signal
                    zd = Variable(zd_data).to(app.DEVICE)
                    zf = Variable(zf_data).to(app.DEVICE)



