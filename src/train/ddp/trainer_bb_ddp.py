# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''
# import warnings
# warnings.filterwarnings("ignore")
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
# from plot.investigation import plot_signal_and_reconstruction
rndm_args = {'mean': 0, 'std': 1}

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

class trainer(object):
    '''Initialize neural network'''
    def __init__(self, gpu, opt):

        """
        Args
        cv  [object]    : content all parsing paramaters from the flag when lauch the python instructions
        opt [dict]      : content all of constant of this models   
        """
        self.opt = opt
        globals().update(self.opt.__dict__)
        super(trainer, self).__init__()
        self.gpu = gpu
        self.rank = self.opt.nr * self.opt.ngpu + self.gpu

        app.logger.debug(f'preparing to train on {self.rank}')
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '9900'
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

        if self.rank == 0:
            self.STEAD(dataset=opt.dataset, dataroot=opt.dataroot, 
                rank = self.rank, batch_size = self.batch_size, 
                nsy = 1280, world_size = opt.world_size)
        dist.barrier()
        self.trn_loader, self.vld_loader = self.STEAD(dataset=opt.dataset, 
                dataroot=opt.dataroot,rank = self.rank, 
                batch_size = self.batch_size, nsy = 1280, world_size = opt.world_size)

        self.Dfnets = []
        self.optzf  = []
        self.oGdxz  = None
        self.oGfxz  = None
        self.oGhxz  = None
        self.dp_mode= True
        
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())]

        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}

        app.logger.debug('DataParallele to be builded ...')
        factory = DataParalleleFactory()
        net = Network(factory)

        if 'broadband' in t:
            self.style='ALICE'

            flagF = True
            n = self.strategy['broadband']

            self.Fef = net.Encoder(self.opt.config['encoder'],self.opt).to(self.rank)
            self.Fef = nn.SyncBatchNorm.convert_sync_batchnorm(self.Fef)

            self.Gdf = net.Decoder(self.opt.config['decoder'], self.opt).to(self.rank)
            self.Gdf = nn.SyncBatchNorm.convert_sync_batchnorm(self.Gdf)

            self.Fef = DDP(self.Fef, device_ids=[self.rank])
            self.Gdf = DDP(self.Gdf, device_ids=[self.rank])

            if self.strategy['tract']['broadband']:
                if None in n:        
                    self.FGf = [self.Fef,self.Gdf]
                    self.oGfxz = reset_net(self.FGf,func=set_weights,
                        lr=glr,b1=b1,b2=b2,
                        weight_decay=0.00001)
                else:
                    print("broadband generators: {0} - {1}".format(*n))
                    
                    self.Fef.load_state_dict(tload(n[0], map_location = map_location)['model_state_dict'])
                    self.Gdf.load_state_dict(tload(n[1], map_location = map_location)['model_state_dict']) 
                    self.oGfxz = Adam(ittc(self.Fef.parameters(),self.Gdf.parameters()),
                                      lr=glr,betas=(b1,b2),weight_decay=0.00001)
                self.optzf.append(self.oGfxz)
                self.Dszf = net.DCGAN_Dz(self.opt.config['Dszf'] , self.opt).to(self.rank)
                self.DsXf = net.DCGAN_Dx(self.opt.config['DsXf'] , self.opt).to(self.rank)
                self.Dfxz = net.DCGAN_DXZ(self.opt.config['Dfxz'], self.opt).to(self.rank)

                self.Dszf = DDP(self.Dszf, device_ids=[self.rank], find_unused_parameters=True)
                self.DsXf = DDP(self.DsXf, device_ids=[self.rank], find_unused_parameters=True)
                self.Dfxz = DDP(self.Dfxz, device_ids=[self.rank], find_unused_parameters=True)

                self.Dfnets.append(self.Dszf)
                self.Dfnets.append(self.DsXf)
                self.Dfnets.append(self.Dfxz)

                self.Dsrzf = net.DCGAN_Dz(self.opt.config['Dsrzf'], self.opt).to(self.rank)
                self.DsrXf = net.DCGAN_Dx(self.opt.config['DsrXf'], self.opt).to(self.rank)

                self.Dsrzf = DDP(self.Dsrzf, device_ids=[self.gpu], find_unused_parameters=True)
                self.DsrXf = DDP(self.DsrXf, device_ids=[self.gpu], find_unused_parameters=True)
                
                self.Dfnets.append(self.DsrXf)
                self.Dfnets.append(self.Dsrzf)

                self.oDfxz = reset_net(self.Dfnets,func=set_weights,lr=rlr,optim='rmsprop')
                self.optzf.append(self.oDfxz)
            else:
                if None not in n:
                    app.logger.debug("broadband generators - no train: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0], map_location = map_location)['model_state_dict'])
                    self.Gdf.load_state_dict(tload(n[1], map_location = map_location)['model_state_dict'])
                else:
                    flagF=False
        self.bce_loss = BCE(reduction='mean')
        self.d_loss = {'Dloss':[0],'Dloss_ali':[0],'Dloss_ali_X':[0],'Dloss_ali_z':[0]} 
        self.g_loss = {'Gloss':[0],'Gloss_ali':[0],'Gloss_ali_X':[0],'Gloss_ali_z':[0],
                        'Gloss_cycle_z':[],'Gloss_cycle_X':[]}

        if self.rank == 0:
            app.logger.debug("Parameters of  Decoders/Decoders ")
            count_parameters(self.FGf)
            app.logger.debug("Parameters of Discriminators ")
            count_parameters(self.Dfnets)
        
    
    def discriminate_filtered_xz(self,Xf,Xfr,zf,zfr):
        DXz = self.Dfxz(zcat(self.Dszf(zfr),self.DsXf(Xf)))
        DzX = self.Dfxz(zcat(self.Dszf(zf),self.DsXf(Xfr)))
        return DXz,DzX
    
    def discriminate_filtered_xx(self,Xf,Xfr):
        Dreal = self.DsrXf(zcat(Xf,Xf ))
        Dfake = self.DsrXf(zcat(Xf,Xfr))
        return Dreal,Dfake

    # @profile
    def discriminate_filtered_zz(self,zf,zfr):
        Dreal = self.Dsrzf(zcat(zf,zf))
        Dfake = self.Dsrzf(zcat(zf,zfr))
        return Dreal,Dfake

    def alice_train_broadband_discriminator_adv_xz(self,Xf,zf):
        zerograd(self.optzf)
        self.Fef.eval(),self.Gdf.eval()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)
        X_gen = self.Gdf(zcat(zf,wnz))
        z_gen = self.Fef(zcat(Xf,wnx)) 
        DXz,DzX = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)

        Dloss_ali = -torch.mean(ln0c(DzX)+ln0c(1.0-DXz))

        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)

        # 2. Generate reconstructions
        X_rec = self.Gdf(z_gen)
        z_rec = self.Fef(X_gen)

        # 3. Cross-Discriminate XX
        Dreal_X,Dfake_X = self.discriminate_filtered_xx(Xf,X_rec)
        Dloss_ali_X     = self.bce_loss(Dreal_X,o1l(Dreal_X))+\
                            self.bce_loss(Dfake_X,o1l(Dfake_X))

        Dreal_z,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)
        Dloss_ali_z     = self.bce_loss(Dreal_z,o1l(Dreal_z))+\
                            self.bce_loss(Dfake_z,o0l(Dfake_z))

        # Total loss
        Dloss = Dloss_ali + 10.*Dloss_ali_X + 100.*Dloss_ali_z

        Dloss.backward()
        self.oDfxz.step(),clipweights(self.Dfnets),zerograd(self.optzf)
        
        self.d_loss['Dloss'].append(Dloss.tolist())
        self.d_loss['Dloss_ali'].append(Dloss_ali.tolist())  
        self.d_loss['Dloss_ali_X'].append(Dloss_ali_X.tolist())  
        self.d_loss['Dloss_ali_z'].append(Dloss_ali_z.tolist())
        
    def alice_train_broadband_generator_adv_xz(self,Xf,zf):
        zerograd(self.optzf)
        self.Fef.train(),self.Gdf.train()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
        l2 = nn.MSELoss()
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)
    
        # 1. Concatenate inputsnex
        X_inp = zcat(Xf,wnx)
        z_inp = zcat(zf,wnz)
        
        # 2. Generate conditional samples
        X_gen = self.Gdf(z_inp)
        z_gen = self.Fef(X_inp)

        DXz,DzX = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)

        # 4. Compute ALI Generator loss
        Gloss_ali = torch.mean(-DXz+DzX)

        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)  
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)
        # 2. Generate reconstructions
        X_rec = self.Gdf(z_gen)
        z_rec = self.Fef(X_gen) 
        # z_rec = latent_resampling(self.Fef(X_gen),nzf,wn1)
 
        # 3. Cross-Discriminate XX
        _,Dfake_X = self.discriminate_filtered_xx(Xf,X_rec)
        Gloss_ali_X = self.bce_loss(Dfake_X,o1l(Dfake_X))
        Gloss_cycle_X = torch.mean(torch.abs(Xf-X_rec))
        
        # 4. Cross-Discriminate ZZ
        Dreal_z,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)

        Gloss_ali_z = self.bce_loss(Dfake_z,o1l(Dfake_z)) +\
                        self.bce_loss(Dreal_z,o1l(Dreal_z))

        # Gloss_cycle_z = torch.mean(torch.abs(zf-z_rec)**2)
        Gloss_cycle_z = l2(zf,z_rec)

        # Total Loss
        Gloss = Gloss_ali +Gloss_ali_X +Gloss_ali_z + Gloss_cycle_X + Gloss_cycle_z
        Gloss.backward(),self.oGfxz.step(),zerograd(self.optzf)
         
        self.g_loss['Gloss'].append(Gloss.tolist()) 
        self.g_loss['Gloss_ali'].append(Gloss_ali.tolist()) 
        self.g_loss['Gloss_ali_X'].append(Gloss_ali_X.tolist())
        self.g_loss['Gloss_ali_z'].append(Gloss_ali_z.tolist())
        self.g_loss['Gloss_cycle_X'].append(Gloss_cycle_X.tolist())
        self.g_loss['Gloss_cycle_z'].append(Gloss_cycle_z.tolist())

    def train_broadband(self):
        globals().update(self.opt.__dict__)

        start_time = time.time()
        for epoch in range(niter):
            for b,batch in enumerate(self.trn_loader):
                xd_data,_,zd_data,*other = batch
                Xf = Variable(xd_data).to(self.gpu, non_blocking = True) # LF-signal
                zf = Variable(zd_data).to(self.gpu, non_blocking = True)
                for _ in range(5):
                    self.alice_train_broadband_discriminator_adv_xz(Xf,zf)
                for _ in range(1):
                    self.alice_train_broadband_generator_adv_xz(Xf,zf)
            str0 = [f'[{epoch+1}/{niter}]']
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-1])))           for k,v in self.g_loss.items()] +['\n']
            str2 = ['\t']+['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-6:-1]))) for k,v in self.d_loss.items()]
            str  = str0+str1+str2
            str = ' | '.join(str)


            if self.rank == 0:
                app.logger.info(str)
                
            if epoch%self.opt.save_checkpoint== 0 and self.gpu == 0:
                app.logger.debug("--- {} minutes ---".format((time.time() - start_time)/60))
                app.logger.info("saving model ...")
                tsave({'epoch':niter,
                        'model_state_dict':self.Fef.state_dict(),
                        'optimizer_state_dict':self.oGfxz.state_dict(),
                        'loss':self.g_loss
                        }, outf+'/network/Fef.pth')
                tsave({'epoch':niter,
                        'model_state_dict':self.Gdf.state_dict(),
                        'optimizer_state_dict':self.oGfxz.state_dict(),
                        'loss':self.g_loss}, outf+'/network/Gdf.pth')    
                tsave({'epoch':niter,
                        'model_state_dict':[Dn.state_dict() for Dn in self.Dfnets],
                        'optimizer_state_dict':self.oDfxz.state_dict(),
                        'loss':self.g_loss}, outf+'/network/DsXz.pth')
            dist.barrier()

        if self.rank == 0:
            app.logger.debug('preparing to plot data ...')
            self.generate()
        dist.destroy_process_group()
    # @profile
    def train(self):
        for t,a in self.strategy['tract'].items():
            # if 'broadband' in t.lower() and a:
            #     self.train_broadband()
            if 'broadband' in t.lower() and a:                    
                self.train_broadband()
            # if 'hybrid' in t.lower() and a:                    
            #     self.train_hybrid()

    # @profile            
    def generate(self):
        globals().update(self.opt.__dict__)
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'broadband' in t and self.strategy['trplt']['broadband']:
            n = self.strategy['broadband']
            if self.rank == 0:
                plt.plot_generate_classic(tag = 'broadband',
                    Qec = self.Fef,
                    Pdc = self.Gdf,
                    opt = self.opt,\
                    trn_set = self.vld_loader,
                    pfx="vld_set_bb",
                    outf=outf)
            if None not in n:
                print("Loading models {} {}".format(n[0],n[1]))
                Fef.load_state_dict(tload(n[0])['model_state_dict'])
                Gdf.load_state_dict(tload(n[1])['model_state_dict'])

        if 'hybrid' in t and self.strategy['trplt']['hybrid']:
            n = self.strategy['hybrid']
            Fef = deepcopy(self.Fef)
            Gdd = deepcopy(self.Gdd)
            Ghz = deepcopy(self.Ghz)
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                Fef.load_state_dict(tload(n[0])['model_state_dict'])
                Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            plt.plot_generate_hybrid(Fef,Gdd,Ghz,device,vtm,\
                                      trn_loader,pfx="trn_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gdd,Ghz,device,vtm,\
                                      tst_loader,pfx="tst_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gdd,Ghz,device,vtm,\
                                      vld_loader,pfx="vld_set_hb",outf=outf)


    def STEAD(self,dataset, dataroot, nsy = 64, batch_size = 64, rank = 0, world_size = 1):
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
