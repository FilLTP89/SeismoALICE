# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from configuration import app
from copy import deepcopy
# from profile_support import profile
from common.common_nn import *
from common.common_torch import * 
import plot.plot_tools as plt
from tools.generate_noise import latent_resampling, noise_generator
from tools.generate_noise import lowpass_biquad
from database.database_sae import random_split 
from tools.leave_p_out import k_folds
from common.ex_common_setup import dataset2loader
from database.database_sae import thsTensorData
import json
# import pprint as pp
import pdb
import os
import logging
# from pytorch_summary import summary
# from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from database.toyset import Toyset
import torch.distributed as dist
import numpy as np
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
    #file: 
    # @profile
    def __init__(self,gpu, opt):


        """
        Intialization part
        """
        self.opt = opt
        globals().update(self.opt.__dict__)
        super(trainer, self).__init__()
        self.gpu = gpu
        self.rank = self.opt.nr * self.opt.ngpu + self.gpu

        torch.cuda.set_device(self.opt.local_rank)
        app.logger.debug(f'preparing to train on {self.rank}')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9910'
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=opt.world_size,
            rank=self.rank)
        seed = 0
        app.logger.debug("init_process_group done ...")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.strategy=strategy
        self.batch_size = opt.batchSize

        """
        Data extraction part
        """
        if self.rank == 0:
            app.logger.debug(f"BatchSize : {self.batch_size}")
            self.STEAD(dataset=opt.dataset, dataroot=opt.dataroot, 
                rank = self.rank, batch_size = self.batch_size, 
                nsy = opt.nsy, world_size = opt.world_size)
        dist.barrier()
        self.trn_loader, self.vld_loader = self.STEAD(dataset=opt.dataset, 
                dataroot=opt.dataroot,rank = self.rank, 
                batch_size = self.batch_size, 
                nsy = opt.nsy, world_size = opt.world_size)

        
        """
        Network Definition part
        """
        self.Dnets = []
        self.optz  = []
        self.oGyx  = None
        self.dp_mode = True

        # glr = 0.0001
        # rlr = 0.0001
        """
        """
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}

        app.logger.debug('DataParallele to be builded ...')
        factory = DataParalleleFactory()
        net = Network(factory)

        if 'unique' in t:
            self.style='ALICE'
            n = self.strategy['unique']
            self.F_  = net.Encoder(opt.config['F'],  opt).to(self.rank)
            self.F_  = nn.SyncBatchNorm.convert_sync_batchnorm(self.F_) 
            
            self.Gy  = net.Decoder(opt.config['Gy'], opt).to(self.rank)
            self.Gy  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Gy)
            
            self.Gx  = net.Decoder(opt.config['Gx'], opt).to(self.rank)
            self.Gx  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Gx)

            self.F_  = DDP(self.F_,device_ids=[self.rank])
            self.Gy  = DDP(self.Gy,device_ids=[self.rank])
            self.Gx  = DDP(self.Gx,device_ids=[self.rank])

            self.FGf  = [self.F_,self.Gy,self.Gx]

            if  self.strategy['tract']['unique']:
                
                if None in n:        
                    self.oGyx = reset_net(self.FGf,
                        func=set_weights,lr=glr,b1=b1,b2=b2,
                        weight_decay=0.00001)
                else:   
                    print("Unique encoder/Multi decoder: {0} - {1}".format(*n))
                    self.F_.load_state_dict(tload(n[0],map_location = map_location)['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1],map_location = map_location)['model_state_dict'])
                    self.Gx.load_state_dict(tload(n[2],map_location = map_location)['model_state_dict'])
                    self.oGyx = Adam(ittc(self.FGf),lr=glr,betas=(b1,b2),weight_decay=0.00001)

                    # self.oGyx = RMSProp(ittc(self.F_.parameters(),
                    #     self.Gy.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,alpha=b2,
                    #     weight_decay=0.00001)
                self.optz.append(self.oGyx)

                ## Discriminators
                # pdb.!()
                # pdb.set_trace()
                self.Dy   = net.DCGAN_Dx(opt.config['Dy'],  opt).to(self.rank)
                self.Dy   = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dy)
                
                self.Dx   = net.DCGAN_Dx(opt.config['Dx'],  opt).to(self.rank)
                self.Dx   = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dx)

                self.Dzb  = net.DCGAN_Dz(opt.config['Dzb'], opt).to(self.rank)
                self.Dzb  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dzb)

                self.Dzf  = net.DCGAN_Dz(opt.config['Dzf'], opt).to(self.rank)
                self.Dzf  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dzf)

                self.Dyz  = net.DCGAN_DXZ(opt.config['Dyz'],opt).to(self.rank)
                self.Dyz  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dyz)

                self.Dxz  = net.DCGAN_DXZ(opt.config['Dxz'],opt).to(self.rank)
                self.Dxz  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dxz)

                self.Dzzb = net.DCGAN_Dz(opt.config['Dzzb'],opt).to(self.rank)
                self.Dzzb = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dzzb)

                self.Dzzf = net.DCGAN_Dz(opt.config['Dzzf'],opt).to(self.rank)
                self.Dzzf = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dzzf)

                self.Dxx  = net.DCGAN_Dx(opt.config['Dxx'], opt).to(self.rank)
                self.Dxx  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dxx)

                self.Dyy  = net.DCGAN_Dx(opt.config['Dyy'], opt).to(self.rank)
                self.Dyy  = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dyy)

                self.Dzyx = net.DCGAN_Dz(opt.config['Dzyx'],opt).to(self.rank)
                self.Dzyx = nn.SyncBatchNorm.convert_sync_batchnorm(self.Dzyx)
                

                self.Dy   = DDP(self.Dy,    device_ids=[self.rank], find_unused_parameters=True)
                self.Dx   = DDP(self.Dx,    device_ids=[self.rank], find_unused_parameters=True)
                self.Dzb  = DDP(self.Dzb,   device_ids=[self.rank], find_unused_parameters=True)
                self.Dzf  = DDP(self.Dzf,   device_ids=[self.rank], find_unused_parameters=True)
                self.Dyz  = DDP(self.Dyz,   device_ids=[self.rank], find_unused_parameters=True)
                self.Dxz  = DDP(self.Dxz,   device_ids=[self.rank], find_unused_parameters=True)
                self.Dzzb = DDP(self.Dzzb,  device_ids=[self.rank], find_unused_parameters=True)
                self.Dzzf = DDP(self.Dzzf,  device_ids=[self.rank], find_unused_parameters=True)
                self.Dxx  = DDP(self.Dxx,   device_ids=[self.rank], find_unused_parameters=True)
                self.Dyy  = DDP(self.Dyy,   device_ids=[self.rank], find_unused_parameters=True)
                self.Dzyx = DDP(self.Dzyx,  device_ids=[self.rank], find_unused_parameters=True)

                self.Dnets.append(self.Dx)
                self.Dnets.append(self.Dy)
                self.Dnets.append(self.Dzb)
                self.Dnets.append(self.Dzf)
                self.Dnets.append(self.Dyz)
                self.Dnets.append(self.Dxz)
                self.Dnets.append(self.Dzzf)
                self.Dnets.append(self.Dzzb)
                self.Dnets.append(self.Dxx)
                self.Dnets.append(self.Dyy)
                self.Dnets.append(self.Dzyx)

                self.oDyxz = reset_net(self.Dnets,
                    func=set_weights,lr=rlr,
                    optim='Adam',b1=b1,b2=b2,
                    weight_decay=0.00001)

                self.optz.append(self.oDyxz)

            else:
                if None not in n:
                    self.F_.load_state_dict(tload(n[0], map_location = map_location)['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1], map_location = map_location)['model_state_dict'])
                    self.Gx.load_state_dict(tload(n[2], map_location = map_location)['model_state_dict'])  
                else:
                    flagF=False
        # pdb.set_trace()
        if self.rank == 0:
            app.logger.debug("Parameters of  Decoders/Decoders ")
            count_parameters(self.FGf)
            app.logger.debug("Parameters of Discriminators ")
            count_parameters(self.Dnets)
        self.bce_loss = BCE(reduction='mean')
        self.g_loss = { 'Gloss':[0],        'Gloss_ali_z':[0],  'Gloss_cycle_xy':[0], 
                        'Gloss_ali_xy':[0], 'Gloss_ind':[0]}

        self.d_loss = { 'Dloss':[0],        'Dloss_ali':[0],    'Dloss_ali_x':[0],  'Dloss_ali_y':[0],
                        'Dloss_ind':[0]}
        
   
       
    # @profile
    def discriminate_xz(self,x,xr,z,zr):
        
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

        return Dxz,Dzx

    def discriminate_yz(self,x,xr,z,zr):
        
        ftz = self.Dzb(zr) #OK : no batchNorm
        ftx = self.Dy(x) #OK : with batchNorm

        zrc = zcat(ftx,ftz)

        ftxz = self.Dyz(zrc) #OK : no batchNorm, don't forget the bias also
        Dxz  = ftxz
       
        
        # Discriminate fake
        ftz = self.Dzb(z)
        ftx = self.Dy(xr)
        zrc = zcat(ftx,ftz)
        ftzx = self.Dyz(zrc)
        Dzx  = ftzx

        return Dxz,Dzx 
    
    # @profile
    def discriminate_xx(self,x,xr):

        Dreal = self.Dxx(zcat(x,x))#with batchNorm
        Dfake = self.Dxx(zcat(x,xr))
        return Dreal,Dfake

    def discriminate_yy(self,x,xr):

        Dreal = self.Dyy(zcat(x,x )) #with batchNorm
        Dfake = self.Dyy(zcat(x,xr))
        return Dreal,Dfake

    # @profile
    def discriminate_zzb(self,z,zr):

        Dreal = self.Dzzb(zcat(z,z )) #no batchNorm
        Dfake = self.Dzzb(zcat(z,zr))
        return Dreal,Dfake

    def discriminate_zzf(self,z,zr):
        # pdb.set_trace()
        Dreal = self.Dzzf(zcat(z,z )) #no batchNorm
        Dfake = self.Dzzf(zcat(z,zr))
        return Dreal,Dfake

    def discriminate_zxy(self,z_yx,z_xy):
        # pdb.set_trace()
        D_zyx = self.Dzyx(z_yx)
        D_zxy = self.Dzyx(z_xy)
        return D_zyx, D_zxy
    
    # @profile
    def alice_train_discriminator_adv(self,y,zd,x,zf):
        # Set-up training        
        zerograd(self.optz)
        modalite(self.FGf, mode = 'eval')
        modalite(self.Dnets, mode ='train')

        # 0. Generate noise
        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,self.gpu,app.RNDM_ARGS)
        wny,wnzd,_ = noise_generator(y.shape,zd.shape,self.gpu,app.RNDM_ARGS)

        
        # 2. Generate conditional samples
        y_gen = self.Gy(zcat(zd,wnzd))
        x_gen = self.Gx(zcat(zf,wnzf))

        # F(y)|y,F(y)|yx,F(y)|x
        zdd_gen,zdf_gen,*other = self.F_(zcat(y,wny)) # zdd_gen = zy, zdf_gen = zyx
        # F(x)|y,F(x)|yx,F(x)|x
        _,zfd_gen, *other = self.F_(zcat(x,wnx)) # zff_gen = zx, zfd_gen = zxy
 
        # [F(y)|yx,F(y)|y]
        zd_gen = zcat(zdf_gen,zdd_gen) # zy generated by y
        # [F(x)|yx,F(x)|x]
        zf_gen = zcat(zfd_gen) # zx generated by x

        
        D_zyx,D_zxy = self.discriminate_zxy(zdf_gen,zfd_gen)

        Dloss_ind = -torch.mean(ln0c(D_zyx)+ln0c(1.0-D_zxy))

        # 3. Cross-Discriminate XZ
        Dyz,Dzy = self.discriminate_yz(y,y_gen,zd,zd_gen)
        Dxz,Dzx = self.discriminate_xz(x,x_gen,zf,zf_gen)

        Dloss_ali_y = -torch.mean(ln0c(Dzy)+ln0c(1.0-Dyz))
        Dloss_ali_x = -torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        Dloss_ali = Dloss_ali_x + Dloss_ali_y

        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,self.gpu,app.RNDM_ARGS)
        wny,wnzd,_ = noise_generator(y.shape,zd.shape,self.gpu,app.RNDM_ARGS)


        y_rec  = self.Gy(zcat(zd_gen,wnzd))
        x_rec  = self.Gx(zcat(zf_gen,wnzf))

        zdd_rec,zdf_rec,*other = self.F_(zcat(y_gen,wny))
        _,zfd_rec,*other = self.F_(zcat(x_gen,wnx))

        zd_rec = zcat(zdf_gen,zdd_gen) # zy generated by y
        zf_rec = zcat(zfd_gen) # zx generated by x


        # 3. Cross-Discriminate XX
        Dreal_y,Dfake_y = self.discriminate_yy(y,y_gen)
        Dreal_x,Dfake_x = self.discriminate_xx(x,x_gen)

        Dloss_rec_y = -torch.mean(ln0c(Dreal_y)+ln0c(1.0-Dfake_y))
        Dloss_rec_x = -torch.mean(ln0c(Dreal_x)+ln0c(1.0-Dfake_x))

        Dreal_zd,Dfake_zd = self.discriminate_zzb(zd,zd_gen)
        Dreal_zf,Dfake_zf = self.discriminate_zzf(zf,zf_gen)

        Dloss_rec_zy = -torch.mean(ln0c(Dreal_zd)+ln0c(1.0-Dfake_zd))
        Dloss_rec_zx = -torch.mean(ln0c(Dreal_zf)+ln0c(1.0-Dfake_zf))

        # Total loss
        Dloss_rec = Dloss_rec_x+Dloss_rec_y+Dloss_rec_zx+Dloss_rec_zy

        Dloss = Dloss_ali + Dloss_ind + Dloss_rec

        Dloss.backward()
        self.oDyxz.step()#,clipweights(self.Dnets),
        zerograd(self.optz)
        self.d_loss['Dloss'].append(Dloss.tolist())  
        self.d_loss['Dloss_ali'].append(Dloss_ali.tolist())  
        self.d_loss['Dloss_ali_x'].append(Dloss_ali_x.tolist())
        self.d_loss['Dloss_ali_y'].append(Dloss_ali_y.tolist())
        self.d_loss['Dloss_ind'].append(Dloss_ind.tolist())
       

    # @profile
    def alice_train_generator_adv(self,y,zd,x,zf,zd_fix, zf_fix):
        # Set-up training
        zerograd(self.optz)
        modalite(self.FGf, mode = 'train')
        modalite(self.Dnets, mode ='train')

        wny,wnzd,_ = noise_generator(y.shape,zd.shape,self.gpu,app.RNDM_ARGS)
        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,self.gpu,app.RNDM_ARGS)
         
        # 2. Generate conditional samples
        y_gen = self.Gy(zcat(zd,wnzd)) #(100,64,64)->(100,3,4096)
        x_gen = self.Gx(zcat(zf,wnzf))

        zdd_gen,zdf_gen,*other = self.F_(zcat(y,wny)) # zdd_gen = zy, zdf_gen = zyx
        _,zfd_gen, *other = self.F_(zcat(x,wnx)) # zff_gen = zx, zfd_gen = zxy
 
        zd_gen = zcat(zdf_gen,zdd_gen) # zy generated by y
        zf_gen = zcat(zfd_gen) # zx generated by x


        # ici il faut les discriminateurs z
        D_zyx,D_zxy = self.discriminate_zxy(zdf_gen,zfd_gen)
        #Dreal_zzy,Dfake_zzy = self.discriminate_zzb(zd,zd_ind)
        #Dreal_zzx,Dfake_zzx = self.discriminate_zzf(zf,zf_ind)
        Gloss_ind = torch.mean(-D_zxy+D_zyx)

        Dyz,Dzy = self.discriminate_yz(y,y_gen,zd,zd_gen)
        Dxz,Dzx = self.discriminate_xz(x,x_gen,zf,zf_gen)

        # 4. Compute ALI Generator loss 
        Gloss_ali = torch.mean(-Dyz+Dzy)+torch.mean(-Dxz+Dzx)
        Gloss_ftm = 0.0

        wny,wnzd,_ = noise_generator(y.shape,zd.shape,self.gpu,app.RNDM_ARGS)
        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,self.gpu,app.RNDM_ARGS)
        
        # 1. Concatenate inputs
        y_gen  = zcat(y_gen,wny)
        x_gen  = zcat(x_gen,wnx) 
        zd_gen = zcat(zd_gen,wnzd) #(100,64,64)
        zf_gen = zcat(zf_gen,wnzf) #(100,32,64)

        # 2. Generate reconstructions
        y_rec = self.Gy(zd_gen)
        x_rec = self.Gx(zf_gen)
        # zd_rec = self.F_(y_gen)[:,:zd.shape[1]]

        zdd_rec,zdf_rec,*other = self.F_(y_gen)
        _,zfd_rec,*other = self.F_(x_gen)

        zd_rec = zcat(zdf_rec,zdd_rec) # zy generated by y
        zf_rec = zcat(zfd_rec) # zx generated by x
    
        # 3. Cross-Discriminate XX
        _,Dfake_y = self.discriminate_yy(y,y_gen[:,:3,:])
        _,Dfake_x = self.discriminate_xx(x,x_gen[:,:3,:])
        # penality 
        penalty = 1.0
        Gloss_ali_xy = torch.mean(Dfake_y + Dfake_x)
        Gloss_cycle_xy = torch.mean(torch.abs(y-y_rec))+torch.mean(torch.abs(x-x_rec))+\
            torch.mean(torch.abs(zd-zd_rec))+torch.mean(torch.abs(zf-zf_rec))
        

        _,Dfake_zd = self.discriminate_zzb(zd,zd_gen[:,:zd.shape[1],:])
        # pdb.set_trace()
        _,Dfake_zf = self.discriminate_zzf(zf,zf_gen[:,:zf.shape[1],:])
        Gloss_ali_z   = torch.mean(Dfake_zd + Dfake_zf)
        Gloss_cycle_z = torch.mean(torch.abs(zd_fix-zd_rec))+torch.mean(torch.abs(zf_fix-zf_rec))

        Gloss = Gloss_ali_xy+Gloss_cycle_xy+Gloss_ali_z+Gloss_ind + Gloss_cycle_z #+Gloss_ftm

        Gloss.backward()
        self.oGyx.step()
        zerograd(self.optz)
         
        self.g_loss['Gloss'].append(Gloss.tolist()) 
        self.g_loss['Gloss_cycle_xy'].append(Gloss_cycle_xy.tolist())
        self.g_loss['Gloss_ali_z'].append(Gloss_ali_z.tolist())
        self.g_loss['Gloss_ali_xy'].append(Gloss_ali_xy.tolist())
        self.g_loss['Gloss_ind'].append(Gloss_ind.tolist())
        
    # @profile
    def train_unique(self):
        app.logger.info('Training on both recorded and synthetic signals ...') 
        globals().update(self.opt.__dict__)
        error = {}

        
        verbose = False

        total_step = len(self.trn_loader)

        app.logger.debug(f"Let's use {self.gpu} GPUs!")

        start_time = time.time()

        for epoch in trange(niter):
            for b,batch in enumerate(self.trn_loader):
            # for b,batch in enumerate(self.trn_loader):
                # pdb.set_trace()
                # y,x,_ = batch
                y, x, zd, zf, *other = batch
                y   = y.to(self.gpu,    non_blocking = True) # recorded signal
                x   = x.to(self.gpu,    non_blocking = True) # synthetic signal
                zd  = zd.to(self.gpu,   non_blocking = True)
                zf  = zf.to(self.gpu,   non_blocking = True)
                # Train G/D
                wnzd = torch.randn(*zd.shape).to(self.gpu, non_blocking = True)
                wnzf = torch.randn(*zf.shape).to(self.gpu, non_blocking = True)
                
                for _ in range(5):
                    self.alice_train_discriminator_adv(y,wnzd,x,wnzf)                 
                for _ in range(1):
                    self.alice_train_generator_adv(y,wnzd,x,wnzf,zd,zf)

                if self.rank == 0:
                    app.logger.debug(f'Epoch [{epoch}/{niter}]\tStep [{b+1}/{total_step}]')


            if self.rank == 0:
                str0 = [f'[{epoch+1}/{niter}]']
                str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-1])))    for k,v in self.g_loss.items()]+['\n']
                str2 = ['\t\t']+['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-6:-1]))) for k,v in self.d_loss.items()]
                str  = str0+str1+str2
                str = ' | '.join(str)
                app.logger.debug(str)

                if epoch%10== 0:
                    app.logger.debug("Time : {:>5.2f} minutes".format((time.time() - start_time)/60))

                if app.LOGGING_LEVEL == logging.DEBUG:
                    GPUtil.showUtilization(all=True)
                
                if epoch%save_checkpoint==0:
                    app.logger.info(f"Saving model at epoch: {epoch}")
                    tsave({'epoch':epoch,'model_state_dict':self.F_.state_dict(),
                           'optimizer_state_dict':self.oGyx.state_dict(),'loss':self.g_loss,},
                           './network/{0}/Fyx_{1}.pth'.format(outf[7:],epoch))
                    tsave({'epoch':epoch,'model_state_dict':self.Gy.state_dict(),
                           'optimizer_state_dict':self.oGyx.state_dict(),'loss':self.g_loss,},
                           './network/{0}/Gy_{1}.pth'.format(outf[7:],epoch))
                    tsave({'epoch':epoch,'model_state_dict':self.Gx.state_dict(),
                           'optimizer_state_dict':self.oGyx.state_dict(),'loss':self.g_loss,},
                           './network/{0}/Gx_{1}.pth'.format(outf[7:],epoch))
        if self.rank == 0:
            app.logger.info('preparing to plot data ...')
            for key, value in self.g_loss.items():
                plt.plot_loss_explicit(losses=value, key=key, outf=outf,niter=niter)

            for key, value in self.d_loss.items():
                plt.plot_loss_explicit(losses=value, key=key, outf=outf,niter=niter)
        dist.destroy_process_group()
                   

    # @profile
    def train(self):
        
        for t,a in self.strategy['tract'].items():
            if 'unique' in t.lower() and a:
                self.train_unique()

    # @profile            
    def generate(self):
        globals().update(self.opt.__dict__)
        app.logger.info("generating result...")
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if self.rank == 0:
            if 'unique' in t and self.strategy['trplt']['unique']:
                plt.plot_generate_classic(tag = 'broadband',
                    Qec     = self.F_,
                    Pdc     = self.Gy,
                    trn_set = self.vld_loader,
                    pfx     ="vld_set_bb_unique",
                    opt     =self.opt,
                    outf    =outf)
                plt.plot_generate_classic(tag = 'filtered',
                    Qec     = self.F_, 
                    Pdc     = self.Gx,
                    trn_set = self.vld_loader,
                    pfx     ="vld_set_fl_unique",
                    opt     =self.opt, 
                    outf    =outf)

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

    def STEAD(self,dataset, dataroot, nsy = 1024, batch_size = 64, rank = 0, world_size = 1):
        ths_trn = torch.load(os.path.join(dataroot,'ths_trn_'+dataset))
        ths_vld = torch.load(os.path.join(dataroot,'ths_vld_'+dataset))
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                ths_trn,
                num_replicas=world_size,
                rank=rank)
        vld_sampler = torch.utils.data.distributed.DistributedSampler(
                ths_vld,
                num_replicas=world_size,
                rank=rank)
        trn_loader = torch.utils.data.DataLoader(dataset=ths_trn, 
                    batch_size =batch_size, 
                    shuffle    =False,
                    num_workers=0,
                    pin_memory =True,
                    sampler    = train_sampler)
        vld_loader = torch.utils.data.DataLoader(dataset=ths_vld, 
                    batch_size =batch_size, 
                    shuffle    =False,
                    num_workers=0,
                    pin_memory =True,
                    sampler    = vld_sampler)
        return trn_loader, vld_loader


