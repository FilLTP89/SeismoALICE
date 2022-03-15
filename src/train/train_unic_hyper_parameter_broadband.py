# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from copy import deepcopy
import imp
import torch
import random
from common.common_nn import *
from common.common_torch import * 
import plot.plot_tools as plt
import profiling.profile_support as profile
from tools.generate_noise import latent_resampling, noise_generator
from database.database_sae import random_split 
from database.database_sae import thsTensorData
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import random 
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

        self.cv         = cv
        self.gr_norm    = []
        self.std        = 1.0
        self.trial      = trial
        self.study_dir  = study_dir
        
        globals().update(cv)
        globals().update(opt.__dict__)
        ngpu_use = torch.cuda.device_count()

        if self.trial!=None:
            self.glr = self.trial.suggest_float("glrx",0.0001, 0.1,log=True)
            self.rlr = self.trial.suggest_float("rlrx",0.0001, 0.1,log=True)
            self.weight_decay = self.trial.suggest_float("weight_decay",1.E-6,1.E-5,log=True)
        else:
            try:
                self.glr = float(opt.config["hparams"]['glry'])
                self.rlr = float(opt.config["hparams"]['rlry'])
                self.weight_decay = float(opt.config["hparams"]["weight_decay"])
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
        ngpu_use = torch.cuda.device_count()
        # To make sure that all operation are deterministic in that GPU for reproductibility
        # torch.backends.cudnn.deterministic  = True
        # torch.backends.cudnn.benchmark      = False

        self.Dnets      = []
        self.optz       = []
        self.oGyx       = None
        # self.dp_mode    = True
        self.losses     = {
            'Dloss':[0],

            'Dloss_ali':[0],
            'Dloss_rec':[0],
            'Dloss_rec_y':[0],
            'Dloss_rec_x':[0],
            'Dloss_rec_zy':[0],
            'Dloss_rec_zxy':[0],

            'Gloss':[0],

            'Gloss_ali':[0],
            'Gloss_cycle_consistency':[0],
            'Gloss_cycle_consistency_y':[0],
            'Gloss_cycle_consistency_zd':[0],
            'Gloss_identity':[0],
            'Gloss_identity_y':[0],
            'Gloss_identity_x':[0],
            'Gloss_identity_zd':[0],
            'Gloss_rec_zx':[0],
            'Gloss_identity_zxy':[0],
            'Gloss_rec_zxy':[0],
            'Gloss_rec_x':[0],
            'Gloss_rec_zf':[0]
        }

        # self.writer_train = SummaryWriter('runs_both/filtered/tuning/training')
        # self.writer_val   = SummaryWriter('runs_both/filtered/tuning/validation')
        # self.writer_debug = SummaryWriter('runs_both/filtered/tuning/debug')
        # self.writer_debug_encoder = SummaryWriter('runs_both/filtered/tuning/debug/encoder')
        
        if self.trial == None:
            self.writer_train = SummaryWriter(f'{opt.summary_dir}/training')
            self.writer_val   = SummaryWriter(f'{opt.summary_dir}/validation')
            self.writer_debug = SummaryWriter(f'{opt.summary_dir}/debug')
        else:
            hparams_dir         = f'{self.study_dir}/hparams/'
            self.writer_hparams = SummaryWriter(f'{hparams_dir}')

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
            # self.Gx  = net.Decoder(opt.config['Gx'], opt)
            self.F_  = nn.DataParallel(self.F_).cuda()
            self.Gy  = nn.DataParallel(self.Gy).cuda()
            # self.Gx  = nn.DataParallel(self.Gx).cuda()

            if  self.strategy['tract']['unique']:
                if None in n:       
                    self.FGf  = [self.F_,self.Gy]
                    self.oGyx = reset_net(self.FGf,
                        func=set_weights,lr=self.glr,b1=b1,b2=b2,
                        weight_decay=self.weight_decay)

                    self.g_scheduler = MultiStepLR(self.oGyx,milestones=[30,80], gamma=0.1) 
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
                    self.F_.load_state_dict(tload(n[0])['model_state_dict'])
                    # self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    # self.Gx.load_state_dict(tload(n[2])['model_state_dict'])

                    app.logger.info("considering master part as pretrained : no grad required")
                    for param in self.F_.module.master.parameters():
                        param.requires_grad = False

                    app.logger.info("considering common part as pretrained : no grad required")
                    for param in self.F_.module.cnn_common.parameters():
                        param.requires_grad = False

                    self.oGyx = MultiStepLR(Adam(ittc(self.F_.parameters()),
                        lr=self.glr,betas=(b1,b2),
                        weight_decay=self.weight_decay))
                    
                    self.g_scheduler = MultiStepLR(self.oGyx,milestones=[30,80], gamma=0.1)
                    # self.oGy = Adam(ittc(self.F_.branch_broadband.parameters(),
                    #     self.Gy.parameters()),
                    #     lr=ngpu_use*glr,betas=(b1,b2),
                    #     weight_decay=0.00001)
                    
                    self.FGf  = [self.F_,self.Gy]
                    app.logger.info("intialization of Gy and the broadband branch for the broadband training ...")
                    self.oGyx = reset_net([self.Gy,self.F_.module.cnn_broadband],
                        func=set_weights,lr=self.glr,b1=b1,b2=b2,
                        weight_decay=self.weight_decay)
                    self.g_scheduler = MultiStepLR(self.oGyx,milestones=[30,80], gamma=0.1)

                    # self.oGyx = RMSProp(ittc(self.F_.parameters(),
                    #     self.Gy.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,alpha=b2,
                    #     weight_decay=0.00001)
                self.optz.append(self.oGyx)
                # self.optz.append(self.oGy)
                self.Dy   = net.DCGAN_Dx( opt.config['Dy'],  opt)
                self.Dzb  = net.DCGAN_Dz( opt.config['Dzb'], opt)
                self.Dyz  = net.DCGAN_DXZ(opt.config['Dyz'], opt)
                self.Dzzb = net.DCGAN_Dz( opt.config['Dzzb'],opt)
                self.Dyy  = net.DCGAN_Dx( opt.config['Dyy'], opt)

                self.Dzyx = net.DCGAN_Dz(opt.config['Dzyx'],opt)
                
                # self.Dx   = net.DCGAN_Dx(opt.config['Dx'],  opt)
                # self.Dzf  = net.DCGAN_Dz(opt.config['Dzf'], opt)
                # self.Dxz  = net.DCGAN_DXZ(opt.config['Dxz'],opt)
                # self.Dzzf = net.DCGAN_Dz(opt.config['Dzzf'],opt)

                self.Dxx  = net.DCGAN_Dx(opt.config['Dxx'], opt)

                self.Dy   = nn.DataParallel(self.Dy  ).cuda()
                self.Dzb  = nn.DataParallel(self.Dzb ).cuda()
                self.Dyz  = nn.DataParallel(self.Dyz ).cuda()
                self.Dzzb = nn.DataParallel(self.Dzzb).cuda()
                self.Dyy  = nn.DataParallel(self.Dyy ).cuda()

                self.Dzyx = nn.DataParallel(self.Dzyx ).cuda()

                self.Dxx  = nn.DataParallel(self.Dxx ).cuda()

                # self.Dzzf = nn.DataParallel(self.Dzzf).cuda()
                

                self.Dnets.append(self.Dy)
                self.Dnets.append(self.Dzb)
                self.Dnets.append(self.Dyz)
                self.Dnets.append(self.Dzzb)
                self.Dnets.append(self.Dyy)
                self.Dnets.append(self.Dzyx)
                self.Dnets.append(self.Dxx)
                # self.Dnets.append(self.Dzzf)

                self.oDyxz = reset_net(self.Dnets,
                    func=set_weights,lr = self.rlr,
                    optim='Adam', b1 = b1, b2 = b2,
                    weight_decay=self.weight_decay)
                
                self.d_scheduler = MultiStepLR(self.oDyxz,milestones=[30,80], gamma=0.1)

                self.optz.append(self.oDyxz)

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
        D_zyx = self.Dzyx(zcat(z_yx,z_yx))
        D_zxy = self.Dzyx(zcat(z_yx,z_xy))
        return D_zyx,D_zxy
    
    # @profile
    def alice_train_discriminator_adv(self,y,zyy,zxy, x):
         # Set-up training        
        zerograd(self.optz)
        modalite(self.FGf,  mode = 'eval')
        modalite(self.Dnets,mode = 'train')
        # 0. Generate noise
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        # 1. Concatenate inputs
        zd_inp  = zcat(zxy,zyy)
        y_inp   = zcat(y,wny)
        
        # 2.1 Generate conditional samples
        y_gen   = self.Gy(zd_inp)
        zyy_F,zyx_F,*other = self.F_(y_inp)
        
        #2.2 Concatenate outputs
        zd_gen= zcat(zyx_F,zyy_F)

        # 3. Cross-Discriminate YZ
        Dyz,Dzy = self.discriminate_yz(y,y_gen,zd_inp,zd_gen)

        # 4. Compute ALI discriminator loss
        Dloss_ali_y = -torch.mean(ln0c(Dzy)+ln0c(1.0-Dyz))

        # 5. Generate reconstructions
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        y_rec       = self.Gy(zd_gen)

        y_gen   = zcat(y_gen,wny)
        zyy_gen,zyx_gen,*other = self.F_(y_gen)
        zd_rec  = zcat(zyx_gen,zyy_gen)

        # 6. Disciminate Cross Entropy  
        Dreal_y,Dfake_y     = self.discriminate_yy(y,y_rec)
        Dloss_rec_y         = self.bce_loss(Dreal_y,o1l(Dreal_y))+\
                                self.bce_loss(Dfake_y,o0l(Dfake_y))
        # Dloss_rec_y       = -torch.mean(ln0c(Dreal_y)+ln0c(1.0-Dfake_y))

        Dreal_zd,Dfake_zd   = self.discriminate_zzb(zd_inp,zd_rec)
        Dloss_rec_zy        = self.bce_loss(Dreal_zd,o1l(Dreal_zd))+\
                                self.bce_loss(Dfake_zd,o0l(Dfake_zd))
        # Dloss_rec_zy        = -torch.mean(ln0c(Dreal_zd)+ln0c(1.0-Dfake_zd)

        # 7. Forcing zxy to equal zyx an zx to equal 0 of the space

        # 7.1 Concatenate inputs
        wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        x_inp   = zcat(x,wnx)
        zf_inp  = zcat(zxy,o0l(zyy))

        # 7.2 Generate samples
        x_gen   = self.Gy(zf_inp)
        zxx_F, zxy_F, *others = self.F_(x_inp)
        
        # 7.3 Concatenate outputs
        zf_gen = zcat(zxy_F,zxx_F)

        # 7.4 Generate reconstructions
        wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        x_gen       = zcat(x_gen,wnx)
        _, zxy_rec,*others = self.F_(x_gen)

        x_rec       = self.Gy(zf_gen)

        # 7.5 Forcing Zxy from X to be guassian
        Dreal_zxy, Dfake_zxy= self.discriminate_zxy(zxy, zxy_rec)
        Dloss_rec_zxy       = self.bce_loss(Dreal_zxy,o1l(Dreal_zxy))+\
                                self.bce_loss(Dfake_zxy,o0l(Dfake_zxy))

        # 7.6 Forcing Zy from X to be useless for the training
        Dreal_x, Dfake_x    = self.discriminate_xx(x,x_rec)
        Dloss_rec_x         = self.bce_loss(Dreal_x,o1l(Dreal_x))+\
                                self.bce_loss(Dfake_x,o0l(Dfake_x))

        # 8. Compute all losses
        Dloss_rec           = Dloss_rec_y + Dloss_rec_zy + Dloss_rec_zxy + Dloss_rec_x
        Dloss_ali           = Dloss_ali_y 
        Dloss               = Dloss_ali   + Dloss_rec

        Dloss.backward()
        self.d_scheduler.step(),
        clipweights(self.Dnets), 
        zerograd(self.optz)

        self.losses['Dloss'].append(Dloss.tolist())
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())

        self.losses['Dloss_rec'   ].append(Dloss_rec.tolist()) 
        self.losses['Dloss_rec_y' ].append(Dloss_rec_y.tolist())
        self.losses['Dloss_rec_x' ].append(Dloss_rec_x.tolist())
        self.losses['Dloss_rec_zy'].append(Dloss_rec_zy.tolist())
        self.losses['Dloss_rec_zxy'].append(Dloss_rec_zxy.tolist())        

    # @profile
    def alice_train_generator_adv(self,y,zyy, zxy, x,epoch, trial_writer=None):
        # Set-up training
        zerograd(self.optz)
        modalite(self.FGf,   mode ='train')
        modalite(self.Dnets, mode ='train')
        
        # 1. Concatenate inputs
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        y_inp   = zcat(y,wny)
        zd_inp   = zcat(zxy,zyy)

        # 2. Generate conditional samples
        y_gen   = self.Gy(zd_inp)
        zyy_F,zyx_F,*others = self.F_(y_inp)

        _zyy_gen= zcat(zyx_F,zyy_F) 
        Dyz,Dzy = self.discriminate_yz(y,y_gen,zd_inp,_zyy_gen)
        
        Gloss_ali =  torch.mean(-Dyz+Dzy) 
        
        # 3. Generate noise
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})

        # 4. Generate reconstructions
        y_rec = self.Gy(_zyy_gen)
        
        y_gen = zcat(y_gen,wny)
        zyy_rec,zyx_rec,*other = self.F_(y_gen)

        zd_rec = zcat(zyx_rec,zyy_rec)
    
        # 5. Cross-Discriminate YY
        _,Dfake_y = self.discriminate_yy(y,y_rec)
        Gloss_cycle_consistency_y   = self.bce_loss(Dfake_y,o1l(Dfake_y))
        Gloss_identity_y            = torch.mean(torch.abs(y-y_rec)) 
        
        # 6. Cross-Discriminate ZZ
        _,Dfake_zd = self.discriminate_zzb(zd_inp,zd_rec)
        Gloss_cycle_consistency_zd  = self.bce_loss(Dfake_zd,o1l(Dfake_zd))
        Gloss_identity_zd           = torch.mean(torch.abs(zd_inp-zd_rec))

        #7. Forcing zxy to equal zyx an zx to equal 0 of the space
        
        # 7.1 Inputs
        wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        x_inp   = zcat(x,wnx)
        zf_inp  = zcat(zxy,o0l(zyy))

        # 7.2 Generate conditional outputs
        zxx_F, zxy_F, *others = self.F_(x_inp)
        zf_gen  = zcat(zxy_F, zxx_F)

        _x_gen   = self.Gy(zf_inp)

        # 7.3 Generate reconstructions values
        wnx,*others = noise_generator(x.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
        x_rec   = self.Gy(zf_gen)

        x_gen   = zcat(_x_gen,wnx)
        zxx_rec, zxy_rec, *others = self.F_(x_gen)

        zf_rec  = zcat(zxy_rec,zxx_rec)

        # 7.4 Forcing Zxy from X to be guassian
        _, Dfake_zxy            = self.discriminate_zxy(zxy, zxy_rec)
        Gloss_identity_zxy      = self.bce_loss(Dfake_zxy, o1l(Dfake_zxy))
        Gloss_rec_zxy           = torch.mean(torch.abs(zxy-zxy_rec)) +\
                                     torch.mean(torch.abs(zxy-zyx_rec))

        # 7.4 Cross-Discriminate XX
        _, Dfake_x              = self.discriminate_xx(x,x_rec)
        Gloss_identity_x        = self.bce_loss(Dfake_x, o1l(Dfake_x))
        Gloss_rec_x             = torch.mean(torch.abs(x - x_rec))

        # 7.5 Forcig Zx to equal 0
        Gloss_rec_zx            = torch.mean(torch.abs(zxx_rec))
        Gloss_rec_zf            = torch.mean(torch.abs(zf_inp-zf_rec))

        # 8. Total Loss
        Gloss_cycle_consistency = Gloss_cycle_consistency_y + Gloss_cycle_consistency_zd 
        Gloss_identity          = ( 
                                    Gloss_identity_y +
                                    Gloss_identity_x +
                                    Gloss_identity_zd + 
                                    Gloss_identity_zxy + 
                                    Gloss_rec_zx + 
                                    Gloss_rec_zf +
                                    Gloss_rec_x + 
                                    Gloss_rec_zxy 
                                )
        Gloss                   = (
                                    Gloss_ali + 
                                    Gloss_cycle_consistency*app.LAMBDA_CONSISTENCY + 
                                    Gloss_identity*app.LAMBDA_IDENTITY
                                )

        if epoch%55 == 0: 
            writer = self.writer_debug if trial_writer == None else trial_writer
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("common/zyx", zyx_rec[idx,:], epoch)
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("common/zxy", zxy_rec[idx,:], epoch)

            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("specific/zyy", zyy_rec[idx,:], epoch)
            for idx in range(opt.batchSize//torch.cuda.device_count()):
                writer.add_histogram("specific/zxx", zxx_rec[idx,:], epoch)


        Gloss.backward()
        self.g_scheduler.step()
        zerograd(self.optz)
         
        self.losses['Gloss'].append(Gloss.tolist())
        self.losses['Gloss_ali'].append(Gloss_ali.tolist())

        self.losses['Gloss_cycle_consistency'   ].append(Gloss_cycle_consistency.tolist())
        self.losses['Gloss_cycle_consistency_y' ].append(Gloss_cycle_consistency_y.tolist())
        self.losses['Gloss_cycle_consistency_zd'].append(Gloss_cycle_consistency_zd.tolist())
        

        self.losses['Gloss_identity'   ].append(Gloss_identity.tolist())
        self.losses['Gloss_identity_y' ].append(Gloss_identity_y.tolist())
        self.losses['Gloss_identity_x' ].append(Gloss_identity_x.tolist())
        self.losses['Gloss_identity_zd'].append(Gloss_identity_zd.tolist())
        
        self.losses['Gloss_identity_zxy'].append(Gloss_identity_zxy.tolist())

        self.losses['Gloss_rec_zx'].append(Gloss_rec_zx.tolist())
        self.losses['Gloss_rec_zxy'].append(Gloss_rec_zxy.tolist())
        self.losses['Gloss_rec_x'].append(Gloss_rec_x.tolist())
        self.losses['Gloss_rec_zf'].append(Gloss_rec_zf.tolist())

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
        bar = trange(opt.niter)

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
                
                for _ in range(5):
                    self.alice_train_discriminator_adv(y,zyy,zyx,x)                 
                for _ in range(1):
                    self.alice_train_generator_adv(y,zyy,zyx,x,epoch, self.writer_histo)
                app.logger.debug(f'Epoch [{epoch}/{opt.niter}]\tStep [{b}/{total_step-1}]')
                
                if epoch%20 == 0 and self.trial== None:
                    for k,v in self.losses.items():
                        self.writer_train.add_scalar('Loss/{}'.format(k),
                            np.mean(np.array(v[-b:-1])),epoch)
            # breakpoint()
            Gloss = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss'][-b:-1])))
            Dloss = '{:>5.3f}'.format(np.mean(np.array(self.losses['Dloss'][-b:-1])))
            Gloss_zxy = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss_identity_zxy'][-b:-1])))
            Dloss_zxy = '{:>5.3f}'.format(np.mean(np.array(self.losses['Dloss_rec_zxy'][-b:-1])))

            bar.set_postfix(Gloss = Gloss, Dloss = Dloss) 
            # bar.set_postfix(Gloss = Gloss, Gloss_zxy = Gloss_zxy, Dloss = Dloss, Dloss_zxy = Dloss_zxy) 
            
            if epoch%25 == 0 and self.trial == None:
                # for k,v in self.losses.items():
                #     self.writer_train.add_scalar('Loss/{}'.format(k),
                #         np.mean(np.array(v[-b:-1])),epoch)
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
                self.writer_val.add_figure('Broadband',figure_bb, epoch)
                self.writer_val.add_figure('Goodness of Fit Broadband',gof_bb, epoch)

                figure_fl, gof_fl = plt.plot_generate_classic(
                        tag     = 'broadband',
                        Qec     = deepcopy(self.F_),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique_hack",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)

                bar.set_postfix(status = 'writing reconstructed filtered signals ...')
                self.writer_val.add_figure('Filtered',figure_fl, epoch)
                self.writer_val.add_figure('Goodness of Fit Filtered',gof_fl, epoch)

                figure_hb, gof_hb = plt.plot_generate_classic(
                        tag     = 'hybrid',
                        Qec     = deepcopy(self.F_),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                
                bar.set_postfix(status = 'writing reconstructed hybrid broadband signals ...')
                self.writer_val.add_figure('Hybrid (Broadband)',figure_hb, epoch)
                self.writer_val.add_figure('Goodness of Fit Hybrid (Broadband)',gof_hb, epoch)
                
                figure_hf, gof_hf = plt.plot_generate_classic(
                        tag     = 'hybrid',
                        Qec     = deepcopy(self.F_),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique_hack",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)

                bar.set_postfix(status = 'writing reconstructed hybrid filtered signals ...')
                self.writer_val.add_figure('Hybrid (Filtered)',figure_hf, epoch)
                self.writer_val.add_figure('Goodness of Fit Hybrid (Filtered)',gof_hf, epoch)

                # if self.trial == None:
                #     #in case of real training
                random.seed(opt.manualSeed)
            
            if (epoch+1)%20 == 0:
                val_accuracy_bb, val_accuracy_fl, val_accuracy_hb = self.accuracy()
                app.logger.info("val_accuracy broadband = {:>5.3f}".format(val_accuracy_bb))
                app.logger.info("val_accuracy filtered  = {:>5.3f}".format(val_accuracy_fl))
                app.logger.info("val_accuracy hybrid    = {:>5.3f}".format(val_accuracy_hb))
                bar.set_postfix(**{'val_accuracy_bb':val_accuracy_bb,
                                'val_accuracy_fl':val_accuracy_fl,
                                'val_accuracy_hb':val_accuracy_hb})
       
                if self.study_dir == None:
                    self.writer_debug.add_scalar('Accuracy/Broadband',val_accuracy_bb, epoch)
                    self.writer_debug.add_scalar('Accuracy/Filtered',val_accuracy_fl, epoch)
                    self.writer_debug.add_scalar('Accuracy/JHybrid',val_accuracy_hb, epoch)

                elif self.study_dir!=None:
                    self.writer_accuracy.add_scalar('Accuracy/Broadband',val_accuracy_bb,epoch)
                    self.writer_accuracy.add_scalar('Accuracy/Filtered',val_accuracy_fl,epoch)
                    self.writer_accuracy.add_scalar('Accuracy/Hybrid',val_accuracy_hb,epoch)
                    
                    self.writer_loss.add_scalar('Loss/Dloss',    float(Dloss),    epoch)
                    self.writer_loss.add_scalar('Loss/Dloss_zxy',float(Dloss_zxy),epoch)
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
                        'model_state_dict'      : self.F_.state_dict(),
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
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dyy.state_dict(),
                        'optimizer_state_dict'  : self.oDyxz.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dyy.pth')
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dzzb.state_dict(),
                        'optimizer_state_dict'  : self.oDyxz.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dzzb.pth')
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
                tsave({ 'epoch'                 : epoch,
                        'model_state_dict'      : self.Dzyx.state_dict(),
                        'optimizer_state_dict'  : self.oDyxz.state_dict(),
                        'loss'                  :self.losses,},
                        root_checkpoint +'/Dzyx.pth')

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

    def accuracy(self):
        total = 0
        EG_h, PG_h  = plt.get_gofs(tag = 'hybrid', 
            Qec = self.F_, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique",
            opt = opt,
            std = self.std, 
            outf = outf)

        EG_b, PG_b  = plt.get_gofs(tag = 'broadband', 
            Qec = self.F_, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique",
            opt = opt,
            std = self.std, 
            outf = outf)

        EG_f, PG_f  = plt.get_gofs(tag = 'broadband', 
            Qec = self.F_, 
            Pdc = self.Gy , 
            trn_set = self.vld_loader, 
            pfx="vld_set_bb_unique_hack",
            opt = opt,
            std = self.std, 
            outf = outf)

        val_h = np.sqrt(np.power([10 - eg for eg in EG_h],2)+\
                np.power([10 - pg for pg in PG_h],2))
        accuracy_hb = val_h.mean().tolist()


        val_b = np.sqrt(np.power([10 - eg for eg in EG_b],2)+\
                     np.power([10 - pg for pg in PG_b],2))
        accuracy_bb = val_b.mean().tolist()

        val_f = np.sqrt(np.power([10 - eg for eg in EG_f],2)+\
                     np.power([10 - pg for pg in PG_f],2))
        accuracy_fl = val_f.mean().tolist()

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
                Qec     = self.F_,
                Pdc     = self.Gy,
                trn_set = self.vld_loader,
                pfx="vld_set_fl_unique",
                opt=opt,
                outf=outf)
            
            # plt.plot_generate_classic(tag = 'filtered',
            #     Qec     = self.F_, 
            #     Pdc     =self.Gx,
            #     trn_set = self.vld_loader,
            #     pfx="vld_set_fl_unique",
            #     opt=opt, outf=outf)

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



