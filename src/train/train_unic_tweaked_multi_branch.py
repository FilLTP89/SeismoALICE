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
from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
import time
import GPUtil
from database.toyset import Toyset, get_dataset
from configuration import app
from tqdm import  tqdm,trange

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
    def __init__(self,cv):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        super(trainer, self).__init__()

        self.cv = cv
        self.gr_norm = []

        globals().update(cv)
        globals().update(opt.__dict__)
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
        self.optz       = []
        self.oGyx       = None
        # self.dp_mode    = True
        self.losses     = {
            'Dloss':[0],
            'Dloss_ali':[0],
            'Dloss_rec':[0],
            'Dloss_rec_y':[0],
            'Dloss_rec_zy':[0],
            'Gloss':[0],
            'Gloss_cycle_consistency':[0],
            'Gloss_cycle_consistency_y':[0],
            'Gloss_cycle_consistency_zd':[0],
            'Gloss_identity':[0],
            'Gloss_identity_y':[0],
            'Gloss_identity_zd':[0],
            'Gloss_ali':[0]
        }

        self.writer_train = SummaryWriter('runs_both/training')
        self.writer_val   = SummaryWriter('runs_both/validation')
        self.writer_debug = SummaryWriter('runs_both/debug')
        self.writer_debug_encoder = SummaryWriter('runs_both/debug/encoder')
        self.writer_debug_decoder = SummaryWriter('runs_both/debug/decoder')


        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        net = Network(DataParalleleFactory())

        self.trn_loader, self.vld_loader = trn_loader, vld_loader

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
                        func=set_weights,lr=ngpu_use*glr,b1=b1,b2=b2,
                        weight_decay=0.00001)
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
                    app.logger.info("Unique encoder/Multi decoder: {0} - {1}".format(*n))
                    checkpoint = tload(n[0])
                    self.start_epoch = checkpoint['epoch']
                    self.losses      = checkpoint['loss']
                    self.F_.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    # self.Gx.load_state_dict(tload(n[2])['model_state_dict'])
                    # self.oGyx = Adam(ittc(self.F_.branch_common.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=ngpu_use*glr,betas=(b1,b2),
                    #     weight_decay=0.00001)
                    # self.oGy = Adam(ittc(self.F_.branch_broadband.parameters(),
                    #     self.Gy.parameters()),
                    #     lr=ngpu_use*glr,betas=(b1,b2),
                    #     weight_decay=0.00001)
                    self.FGf  = [self.F_,self.Gy]
                    self.oGyx = reset_net(self.FGf,
                        func=set_weights,lr=ngpu_use*glr,b1=b1,b2=b2,
                        weight_decay=0.00001)

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

                # self.Dzyx = net.DCGAN_Dz(opt.config['Dzyx'],opt)
                # self.Dx   = net.DCGAN_Dx(opt.config['Dx'],  opt)
                # self.Dzf  = net.DCGAN_Dz(opt.config['Dzf'], opt)
                # self.Dxz  = net.DCGAN_DXZ(opt.config['Dxz'],opt)
                # self.Dzzf = net.DCGAN_Dz(opt.config['Dzzf'],opt)
                # self.Dxx  = net.DCGAN_Dx(opt.config['Dxx'], opt)

                self.Dy   = nn.DataParallel(self.Dy).cuda()
                self.Dzb  = nn.DataParallel(self.Dzb).cuda()
                self.Dyz  = nn.DataParallel(self.Dyz).cuda()
                self.Dzzb = nn.DataParallel(self.Dzzb).cuda()
                self.Dyy  = nn.DataParallel(self.Dyy).cuda()
                

                self.Dnets.append(self.Dy)
                self.Dnets.append(self.Dzb)
                self.Dnets.append(self.Dyz)
                self.Dnets.append(self.Dzzb)
                self.Dnets.append(self.Dyy)

                self.oDyxz = reset_net(self.Dnets,
                    func=set_weights,lr=ngpu_use*rlr,
                    optim='Adam',b1=b1,b2=b2,
                    weight_decay=0.00001)

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
        print("Parameters of Discriminators ")
        count_parameters(self.Dnets)
        
        
       
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
    def alice_train_discriminator_adv(self,y,zyy,zxy, x =  None):
        # Set-up training        
        zerograd(self.optz)
        modalite(self.FGf,  mode = 'eval')
        modalite(self.Dnets,mode = 'train')
        
        # 0. Generate noise
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.RNDM_ARGS)
        # 1. Concatenate inputs
        z_inp = zcat(zxy,zyy)
        y_inp = zcat(y,wny)
        
        # 2.1 Generate conditional samples
        y_gen = self.Gy(z_inp)
        zyy_F,zyx_F,*other = self.F_(y_inp)
        #2.2 Concatanete outputs
        zyy_gen = zcat(zyx_F,zyy_F)

        # 3. Cross-Discriminate YZ
        Dyz,Dzy = self.discriminate_yz(y,y_gen,z_inp,zyy_gen)

        # 4. Compute ALI discriminator loss
        Dloss_ali_y = -torch.mean(ln0c(Dzy)+ln0c(1.0-Dyz))
        
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.RNDM_ARGS)

        # 5. Generate reconstructions
        y_rec  = self.Gy(zyy_gen)
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
        
        # 7. Compute all losses
        Dloss_rec           = Dloss_rec_y + Dloss_rec_zy
        Dloss_ali           = Dloss_ali_y 
        Dloss               = Dloss_ali   + Dloss_rec

        Dloss.backward()
        self.oDyxz.step(), clipweights(self.Dnets), zerograd(self.optz)

        self.losses['Dloss'].append(Dloss.tolist())
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())

        self.losses['Dloss_rec'   ].append(Dloss_rec.tolist()) 
        self.losses['Dloss_rec_y' ].append(Dloss_rec_y.tolist())
        self.losses['Dloss_rec_zy'].append(Dloss_rec_zy.tolist())


    # @profile
    def alice_train_generator_adv(self,y,zyy, zxy,\
                x = None, zyy_fix = None, zxy_fix= None,epoch = None):
        # Set-up training
        zerograd(self.optz)
        modalite(self.FGf,   mode ='train')
        modalite(self.Dnets, mode ='train')

        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.RNDM_ARGS)
        
        # 1. Concatenate inputs
        y_inp  = zcat(y,wny)
        z_inp  = zcat(zxy,zyy)
         
        # 2. Generate conditional samples
        y_gen = self.Gy(z_inp)
        zyy_F,zyx_F,*other = self.F_(y_inp) 
 
        zyy_gen = zcat(zyx_F,zyy_F) 
        Dyz,Dzy = self.discriminate_yz(y,y_gen,z_inp,zyy_gen)
        
        Gloss_ali =  torch.mean(-Dyz+Dzy) 
        
        # 3. Generate noise
        wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.RNDM_ARGS)

        # 4. Generate reconstructions
        y_rec = self.Gy(zyy_gen)
        y_gen = zcat(y_gen,wny)
        zyy_F,zyx_F,*other = self.F_(y_gen)
        zyy_rec = zcat(zyx_F,zyy_F)
    
        # 5. Cross-Discriminate XX
        _,Dfake_y = self.discriminate_yy(y,y_rec)
        Gloss_cycle_consistency_y   = self.bce_loss(Dfake_y,o1l(Dfake_y))
        Gloss_identity_y            = torch.mean(torch.abs(y-y_rec)**2) 
        
        # 6. Cross-Discriminate ZZ
        _,Dfake_zd = self.discriminate_zzb(z_inp,zyy_rec)
        Gloss_cycle_consistency_zd  = self.bce_loss(Dfake_zd,o1l(Dfake_zd))
        Gloss_identity_zd           = torch.mean(torch.abs(z_inp-zyy_rec)**2)

        # 7. Total Loss
        Gloss_cycle_consistency     = Gloss_cycle_consistency_y +  Gloss_cycle_consistency_zd
        Gloss_identity              = Gloss_identity_y + Gloss_identity_zd
        Gloss                       = Gloss_ali + Gloss_cycle_consistency + Gloss_identity
                
        if epoch == 10:
            for batch in range(opt.batchSize):
                self.writer_debug.add_histogram("zyy_rec",zyy_rec[batch,:], epoch)
                self.writer_debug.add_histogram("zxy",zxy[batch,:],epoch)

        Gloss.backward()
        self.oGyx.step()
        zerograd(self.optz)
         
        self.losses['Gloss'].append(Gloss.tolist())
        self.losses['Gloss_ali'].append(Gloss_ali.tolist())

        self.losses['Gloss_cycle_consistency'   ].append(Gloss_cycle_consistency.tolist())
        self.losses['Gloss_cycle_consistency_y' ].append(Gloss_cycle_consistency_y.tolist())
        self.losses['Gloss_cycle_consistency_zd'].append(Gloss_cycle_consistency_zd.tolist())

        self.losses['Gloss_identity'   ].append(Gloss_identity.tolist())
        self.losses['Gloss_identity_y' ].append(Gloss_identity_y.tolist())
        self.losses['Gloss_identity_zd'].append(Gloss_identity_zd.tolist())

        

    def generate_latent_variable(self, batch, nch_zd,nzd, nch_zf = 128,nzf = 128):
        zyy  = torch.randn(*[batch,nch_zd,nzd]).to(app.DEVICE, non_blocking = True)
        zxx  = torch.randn(*[batch,nch_zd,nzd]).to(app.DEVICE, non_blocking = True)

        zyx  = torch.randn(*[batch,nch_zf,nzf]).to(app.DEVICE, non_blocking = True)
        zxy  = torch.randn(*[batch,nch_zf,nzf]).to(app.DEVICE, non_blocking = True)
        return zyy, zyx, zxx, zxy

    # @profile
    def train_unique(self):
        app.logger.info('Training on both recorded and synthetic signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)

        total_step = len(self.trn_loader)
        app.logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")

        bar = trange(self.start_epoch, niter+1)
        nch_zd, nzd = 24,128
        nch_zf, nzf = 8,128
        
        for epoch in bar:
            for b,batch in enumerate(self.trn_loader):
                y, *others = batch
                # y   = y.to(app.DEVICE, non_blocking = True)
                y   = y.to(app.DEVICE, non_blocking = True)
    
                # getting noise shape
                zyy,zyx, *other = self.generate_latent_variable(
                            batch   = opt.batchSize,
                            nzd     = nzd,
                            nch_zd  = nch_zd,
                            nzf     = nzf,
                            nch_zf  = nch_zf)
                for _ in range(5):
                    self.alice_train_discriminator_adv(y,zyy,zyx)                 
                for _ in range(1):
                    self.alice_train_generator_adv(y,zyy,zyx, epoch=epoch)
                app.logger.debug(f'Epoch [{epoch}/{opt.niter}]\tStep [{b}/{total_step-1}]')

            if epoch%10== 0:
                torch.manual_seed(100)
                for k,v in self.losses.items():
                    self.writer_train.add_scalar('Loss/{}'.format(k),
                        np.mean(np.array(v[-b:-1])),epoch)

                figure_bb, gof_bb = plt.plot_generate_classic(
                        tag     = 'broadband',
                        Qec     = deepcopy(self.F_),
                        Pdc     = deepcopy(self.Gy),
                        trn_set = self.vld_loader,
                        pfx     ="vld_set_bb_unique",
                        opt     = opt,
                        outf    = outf, 
                        save    = False)
                self.writer_val.add_figure('Broadband',figure_bb,epoch)
                self.writer_val.add_figure('Goodness of Fit',gof_bb,epoch)
             
            Gloss = '{:>5.3f}'.format(np.mean(np.array(self.losses['Gloss'][-b:-1])))
            Dloss = '{:>5.3f}'.format(np.mean(np.array(self.losses['Dloss'][-b:-1])))
            
            bar.set_postfix(Gloss=Gloss, Dloss = Dloss)
           
            if epoch%save_checkpoint == 0:
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

        
        for key, value in self.losses.items():
            plt.plot_loss_explicit(losses=value, key=key, outf=outf,niter=niter)
                           
    

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
                pfx="vld_set_bb_unique",
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



