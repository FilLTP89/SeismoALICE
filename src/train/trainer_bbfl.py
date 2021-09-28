# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
# from profile_support import profile
# from profile_support import profile
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
# from pytorch_summary import summary
# from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from database.toyset import Toyset
from configuration import app
# from laploss import *
# from pl_bolts.datamodules.async_dataloader import AsynchronousLoader
# from pytorchsummary.torchsummary import summary
# import numpy as np

# app.RNDM_ARGS = {'mean': 0, 'std': 1.0}

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
# b1 = 0.5
# b2 = 0.9999

b1 = 0.5
b2 = 0.9999

nch_tot = 3
penalty_wgangp = 10.
nly = 5

class trainer(object):
    '''Initialize neural network'''
    # @profile
    def __init__(self,cv):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        super(trainer, self).__init__()
    
        self.cv = cv
        self.gr_norm = []

        # define as global variable the cv object. 
        # And therefore this latter is become accessible to the methods in this class
        globals().update(cv)
        # define as global opt and passing it as a dictonnary here
        globals().update(opt.__dict__)

        dataset = Toyset(nsy =opt.nsy)
    
        train_set, vld_set = torch.utils.data.random_split(dataset, [80,20])
        self.trn_loader = torch.utils.data.DataLoader(dataset=train_set, 
                    batch_size=opt.batchSize, shuffle=True)
        self.vld_loader = torch.utils.data.DataLoader(dataset=vld_set, 
                    batch_size=opt.batchSize, shuffle=True)

        # passing the content of file ./strategy_bb_*.txt
        self.strategy=strategy
        self.ngpu = ngpu
        # dist.init_process_group("gloo", rank=ngpu, world_size=1)

        nzd = opt.nzd
        ndf = opt.ndf
        
        # the follwings variable are the instance for the object Module from 
        # the package pytorch: torch.nn.modulese. 
        # the names of variable are maped to the description aforementioned
        self.Fed = Module()
        self.Gdd = Module()
        self.DsXd = Module()
        self.Dszd = Module()
        self.DsXf = Module()
        self.Dszf = Module()

        self.Ddnets = []
        self.Dfnets = []
        self.Dhnets = []
        self.optzd  = []
        self.optzf  = []
        self.optzh  = []
        self.oGdxz=None
        self.oGfxz=None
        self.oGhxz=None
        self.dp_mode = True
        
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        """
            This part is for training with the broadband signal
        """
        # pdb.set_trace()
        #we are getting number of cpus 
        cpus  =  int(os.environ.get('SLURM_NPROCS'))
        #we determine in which kind of environnement we are 
        if(cpus ==1 and opt.ngpu >=1):
            print('ModelParallele to be builded ...')
            self.dp_mode = False
            factory = ModelParalleleFactory()
        elif(cpus >1 and opt.ngpu >=1):
            print('DataParallele to be builded ...')
            factory = DataParalleleFactory()
            self.dp_mode = True
        else:
            print('environ not found')
        net = Network(factory)

        # pdb.set_trace()

        if 'broadband' in t:
            self.style='ALICE'
            # act = acts[self.style]
            flagT = True
            
            n = self.strategy['broadband']
            print("Loading broadband generators")

            # pdb.set_trace()
            # # Encoder broadband Fed
            self.Fed = net.Encoder(opt.config["encoder"],opt)
            # # Decoder broadband Gdd
            self.Gdd = net.Decoder(opt.config["decoder"],opt)
            
            #if we training with the broadband signal
                # we read weigth and bias if is needed and then we set-up the Convolutional Neural 
                # Network needed for the Discriminator
            if self.strategy['tract']['broadband']:
                if None in n:
                    self.FGd = [self.Fed,self.Gdd]
                    self.oGdxz = reset_net(self.FGd,func=set_weights,lr=glr,b1=b1,b2=b2,\
                            weight_decay=None)
                else:
                    print("Broadband generators: {0} - {1}".format(*n))
                    self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                    self.Fed.to(torch.float32)
                    self.Gdd.to(torch.float32)
                    self.oGdxz = Adam(ittc(self.Fed.parameters(),self.Gdd.parameters()),
                                      lr=glr,betas=(b1,b2))#,weight_decay=None)
                self.optzd.append(self.oGdxz)
                # create a conv1D of 2 layers for the discriminator to tranfrom from (,,32) to (,,512)

                self.Dszd = net.DCGAN_Dz(opt.config['Dszd'],opt)
                self.DsXd = net.DCGAN_Dx(opt.config['DsXd'],opt)
                self.Ddxz = net.DCGAN_DXZ(opt.config['Ddxz'],opt)

                # self.Ddnets.append(self.Fed)
                # self.Ddnets.append(self.Gdd)
                self.Ddnets.append(self.DsXd)  
                self.Ddnets.append(self.Dszd)
                self.Ddnets.append(self.Ddxz)
                self.oDdxz = reset_net(self.Ddnets,func=set_weights,lr=rlr,b1=b1,b2=b2)
                self.optzd.append(self.oDdxz)



                # pdb.set_trace()
                # self.DsXf = net.DCGAN_Dx(opt.app['DsXf'], opt)
                # self.Dszf = net.DCGAN_Dz(opt.app['Dszf'], opt)
                # self.Dfxz = net.DCGAN_DXZ(opt.app['Dfxz'], opt)
            else:
                if None not in n:
                    print("Broadband generators - NO TRAIN: {0} - {1}".format(*n))
                    self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                else:
                    flagT=False

        if 'filtered' in t:
            self.style='ALICE'
            # act = acts[self.style]
            flagF = True
            n = self.strategy['filtered']
            print("Loading filtered generators")
            # pdb.set_trace()
            self.Fef = net.Encoder(opt.config['encoder'], opt)
            self.Gdf = net.Decoder(opt.config['decoder'], opt)

            if self.strategy['tract']['filtered']:
                if None in n:        
                    self.FGf = [self.Fef,self.Gdf]
                    self.oGfxz = reset_net(self.FGf,func=set_weights,lr=glr,b1=b1,b2=b2,
                            weight_decay=0.00001)
                else:
                    print("Filtered generators: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdf.load_state_dict(tload(n[1])['model_state_dict'])    
                    self.oGfxz = Adam(ittc(self.Fef.parameters(),self.Gdf.parameters()),
                                      lr=glr,betas=(b1,b2),weight_decay=0.00001)
                self.optzf.append(self.oGfxz)
                # pdb.set_trace()
                self.Dszf = net.DCGAN_Dz(opt.config['Dszf']  , opt)
                self.DsXf = net.DCGAN_Dx(opt.config['DsXf']  , opt)
                self.Dfxz = net.DCGAN_DXZ(opt.config['Dfxz'] , opt)

                # pdb.set_trace()
                # if n[2]:
                #     self.DsXf.load_state_dict(tload(n[2])['model_state_dict'])

                self.Dfnets.append(self.DsXf)
                self.Dfnets.append(self.Dszf)
                self.Dfnets.append(self.Dfxz)

                self.Dsrzf =  net.DCGAN_Dz(opt.config['Dsrzf'], opt)
                self.DsrXf =  net.DCGAN_Dx(opt.config['DsrXf'], opt)
                

                self.Dfnets.append(self.DsrXf)
                self.Dfnets.append(self.Dsrzf)

                self.oDfxz = reset_net(self.Dfnets,func=set_weights,lr=rlr,optim='rmsprop')
                self.optzf.append(self.oDfxz)

                # pdb.set_trace()

            else:
                if None not in n:
                    print("Filtered generators - no train: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdf.load_state_dict(tload(n[1])['model_state_dict'])  
                else:
                    flagF=False

        """
            This part is for the training with the hybrid signal
        """
        if 'hybrid' in t and flagF and flagT:
            self.style='WGAN'
            # act = acts[self.style]
            n = self.strategy['hybrid']
            print("Loading hybrid generators")

            if hasattr(opt.app,'Ghz'):
                nlayers['Gdf'] = opt.config['decoder']['nlayers']
                kernels['Gdf'] = opt.config['decoder']['kernel']
                strides['Gdf'] = opt.config['decoder']['strides']
                padding['Gdf'] = opt.config['decoder']['padding']
                outpads['Gdf'] = opt.config['decoder']['outpads']
            else:
                print('!!! warnings no app found\n\tassuming default parameters for the hybrid generator')

            self.Ghz = Encoder(ngpu=ngpu,dev=app.DEVICE,nz=nzd,nzcl=0,
                               nch=2*nzf,ndf=nzf*4,szs=32,nly=nlayers['Ghz'],
                               ker=kernels['Ghz'],std=strides['Ghz'],
                               pad=padding['Ghz'],dil=1,grp=1,dpc=0.0,
                               bn=True,act=act['Ghz']).to(app.DEVICE)
            if None not in n:
                print("Hybrid generators - no train: {0} - {1} - {2}".format(*n))
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            if self.strategy['tract']['hybrid'] or self.strategy['trdis']['hybrid']: 
                if self.style=='WGAN':
                    self.oGhxz = reset_net([self.Ghz],func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    self.oGhxz = reset_net([self.Ghz],func=set_weights,lr=glr,b1=b1,b2=b2)
                self.optzh.append(self.oGhxz)

                if hasattr(opt.config,'Dsrzd'):
                    self.Dsrzd = DCGAN_DXZ(ngpu=ngpu,nc=2*nzd,n_extra_layers=opt.config['Dsrzd']['layers'],dpc=0.25,
                                           activation=act['Ddxz'],wf=False).to(app.DEVICE)
                else:
                    self.Dsrzd = DCGAN_DXZ(ngpu=ngpu,nc=2*nzd,n_extra_layers=2,dpc=0.25,
                                           activation=act['Ddxz'],wf=False).to(app.DEVICE)
                
                if self.style=='WGAN':
                    if hasattr(opt.config,'DsrXd'):
                        self.DsrXd = Encoder(ngpu=ngpu,dev=app.DEVICE,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=opt.config['DsrXd']['kernels'],\
                                     std=opt.config['DsrXd']['strides'],\
                                     pad=opt.config['DsrXd']['padding'],\
                                     dil=opt.config['DsrXd']['dilation'],\
                                     grp=1,dpc=0.25,bn=False,\
                                     act=act['DhXd']).to(app.DEVICE)
                    else:
                        self.DsrXd = Encoder(ngpu=ngpu,dev=app.DEVICE,nz=1,nzcl=0,nch=2*nch_tot,
                                         ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                         pad=1,dil=1,grp=1,dpc=0.25,bn=False,\
                                         act=act['DhXd']).to(app.DEVICE)
                        print('!!! warnings no configuration found for DsrXd')
                    self.Dhnets.append(self.Dsrzd)
                    self.Dhnets.append(self.DsrXd)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    if hasattr(opt.config,'DsrXd'):
                        self.DsrXd = Encoder(ngpu=ngpu,dev=app.DEVICE,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=op.config['DsrXd']['nlayers'],\
                                     ker=opt.config['DsrXd']['kernel'],\
                                     std=opt.config['DsrXd']['strides'],\
                                     pad=opt.config['DsrXd']['padding'],\
                                     dil=opt.config['DsrXd']['dilation'],\
                                     grp=1,dpc=0.25,bn=True,\
                                     act=act['DhXd']).to(app.DEVICE)    
                    else:
                        self.DsrXd = Encoder(ngpu=ngpu,dev=app.DEVICE,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=True,\
                                     act=act['DhXd']).to(app.DEVICE)
                    self.Dhnets.append(self.Dsrzd)
                    self.Dhnets.append(self.DsrXd)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,b1=b1,b2=b2)
                self.optzh.append(self.oDhzdzf) 
        # Loss Criteria
        # pdb.set_trace()
        self.bce_loss = BCE(reduction='mean').to(ngpu-1)
        self.losses = {'Dloss_t':[0],'Dloss_f':[0],'Dloss_t':[0],
                       'Gloss_x':[0],'Gloss_z':[0],'Gloss_t':[0],
                       'Gloss_xf':[0],'Gloss_xt':[0],'Gloss_f':[0],
                       'Gloss':[0],'Dloss':[0],'Gloss_ftm':[0],'Gloss_ali_X':[0],
                       'Gloss_ali_z':[0],'Gloss_cycle_X':[0],
                       'Gloss_cycle_z':[0],'Dloss_ali':[0],
                       'Dloss_ali_X':[0],'Dloss_ali_z':[0], 'norm_grad':[0]}
        
        #end of constructior

    ''' Methode that discriminate real and fake signal for broadband type '''
    # @profile
    def discriminate_broadband_xz(self,Xd,Xdr,zd,zdr):
        # pdb.set_trace()
        a = self.DsXd(Xd)
        b = self.Dszd(zdr)
        
        zrc = zcat(a,b)
       
        DXz = self.Ddxz(zrc)
        
        c = self.DsXd(Xdr)
        d = self.Dszd(zd)
        # print("\t||DsXd(Xdr) : ",c.shape,"\tDszd(zd) : ", d.shape)
        zrc = zcat(c,d)
        DzX = self.Ddxz(zrc)

        return DXz,DzX
        #end of discriminate_broadband_xz function

    ''' Methode that discriminate real and fake signal for filtred type '''
    def discriminate_filtered_xz(self,Xf,Xfr,zf,zfr):
        # pdb.set_trace()
        # Discriminate real
        ftz = self.Dszf(zfr)
        ftX = self.DsXf(Xf)
        zrc = zcat(ftX[0],ftz[0])
        ftr = ftz[1]+ftX[1]
        ftXz = self.Dfxz(zrc)
        DXz  = ftXz[0]
        ftr += ftXz[1]
        
        # Discriminate fake
        ftz = self.Dszf(zf)
        ftX = self.DsXf(Xfr)
        zrc = zcat(ftX[0],ftz[0])
        ftf = ftz[1]+ftX[1]
        ftzX = self.Dfxz(zrc)
        DzX  = ftzX[0]
        ftf += ftzX[1]
        
        return DXz,DzX,ftr,ftf
        
        # return DXz,DzX,ftr,ftf
    
    ''' Methode that discriminate real and fake hybrid signal type'''
    def discriminate_hybrid_xz(self,Xd,Xdr,zd,zdr):
        
        # Discriminate real
        ftz = self.Dszd(zdr)
        ftX = self.DsXd(Xd)
        zrc = zcat(ftX[0],ftz[0])
        ftr = ftz[1]+ftX[1]
        ftXz = self.Dhzdzf(zrc)
        DXz  = ftXz[0]
        ftr += ftXz[1]
        
        # Discriminate fake
        ftz = self.Dszd(zd)
        ftX = self.DsXd(Xdr)
        zrc = zcat(ftX[0],ftz[0])
        ftf = ftz[1]+ftX[1]
        ftzX = self.Dhzdzf(zrc)
        DzX  = ftzX[0]
        ftf += ftzX[1]
        
        return DXz,DzX,ftr,ftf

    # @profile
    def discriminate_filtered_xx(self,Xf,Xfr):
        # pdb.set_trace()
        Dreal = self.DsrXf(zcat(Xf,Xf ))
        Dfake = self.DsrXf(zcat(Xf,Xfr))
        return Dreal,Dfake

    # @profile
    def discriminate_filtered_zz(self,zf,zfr):
        # pdb.set_trace()
        Dreal = self.Dsrzf(zcat(zf,zf ))
        Dfake = self.Dsrzf(zcat(zf,zfr))
        return Dreal,Dfake

    # @profile
    def discriminate_hybrid_xx(self,Xf,Xfr):
        Dreal = self.DsrXd(zcat(Xf,Xf ))
        Dfake = self.DsrXd(zcat(Xf,Xfr))
        return Dreal,Dfake

    # @profile
    def discriminate_hybrid_zz(self,zf,zfr):
        Dreal = self.Dsrzd(zcat(zf,zf ))
        Dfake = self.Dsrzd(zcat(zf,zfr))
        return Dreal,Dfake

    ####################
    ##### CLASSIC  #####
    ####################
    # @profile
    def alice_train_broadband_discriminator_explicit_xz(self,Xd,zd):
        zerograd(self.optzd)
        self.Fed.eval(),self.Gdd.eval()
        self.DsXd.train(),self.Dszd.train(),self.Ddxz.train()
        
        # pdb.set_trace()
        # 0. Generate noise
        place = opt.dev if self.dp_mode else ngpu-1
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)


        # 1. Concatenate inputs
        wnx = wnx.to(Xd.app.DEVICE)
        wnz = wnz.to(zd.app.DEVICE)

        X_inp = zcat(Xd,wnx)
        z_inp = zcat(zd,wnz)

        # 2. Generate conditional samples 1 time
        X_gen = self.Gdd(z_inp)
        z_gen = self.Fed(X_inp)
        # pdb.set_trace()
        # X_gen = self.Gdd(zcat(wnz,self.Fed(X_inp)))
        # z_gen = self.Fed(zcat(wnx,self.Gdd(z_inp)))
        # torch.cuda.empty_cache()
        
        # 3. Cross-Discriminate XZ
        Dxz,Dzx = self.discriminate_broadband_xz(Xd,X_gen,zd,z_gen)

        # Dloss_ali = -(torch.mean(Dxz)-torch.mean(Dzx))
        Dloss_ali = -torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        
        # gr_norm = self.get_Dgrad_norm(Xd, X_gen, zd, z_gen)

        # Dgrad_loss = ((gr_norm - 1.) ** 2).mean().to(ngpu-1)
        
        # self.losses["norm_grad"].append(Dgrad_loss.tolist())
            
        # Dloss_ali+= penalty_wgangp * Dgrad_loss
        # Total loss
        Dloss = Dloss_ali.to(place)
        # print("\t||Dloss :", Dloss)
        Dloss.backward()
        # torch.cuda.empty_cache()
        self.oDdxz.step()
        zerograd(self.optzd)
        self.losses['Dloss_t'].append(Dloss.tolist())
        # torch.cuda.empty_cache()
        # GPUtil.showUtilization(all=True)
    
    @profile
    def alice_train_broadband_generator_explicit_xz(self,Xd,zd):
        # pdb.set_trace()
        zerograd(self.optzd)
        self.Fed.train(),self.Gdd.train()
        self.DsXd.train(),self.Dszd.train(),self.Ddxz.train()
        # 0. Generate noise
        place = opt.dev if self.dp_mode else ngpu-1
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
        #Put wnx and wnz and the same app.DEVICE of X_inp and z_inp
        wnx = wnx.to(Xd.app.DEVICE)
        wnz = wnz.to(zd.app.DEVICE)
        # 1. Concatenate inputs
        X_inp = zcat(Xd,wnx)
        z_inp = zcat(zd,wnz)
        
        # pdb.set_trace()
        # 2. Generate conditional samples
        X_gen = self.Gdd(z_inp)
        z_gen = self.Fed(X_inp)
        # X_gen = self.Gdd(zcat(wnz,self.Fed(X_inp)))
        # z_gen = self.Fed(zcat(wnx,self.Gdd(z_inp)))
        # z_gen = latent_resampling(self.Fed(X_inp),nzd,wn1)
        
        # 3. Cross-Discriminate XZ
        # pdb.set_trace()
        Dxz,Dzx = self.discriminate_broadband_xz(Xd,X_gen,zd,z_gen)
        # 4. Compute ALI Generator loss WGAN
        Gloss_ali = torch.mean(-Dxz +Dzx).to(app.DEVICE)
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
        # # passign values to the CPU 0
        
        wnx = wnx
        wnz = wnz
        wn1 = wn1
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx.to(X_gen.app.DEVICE))    
        z_gen = zcat(z_gen,wnz.to(z_gen.app.DEVICE))
        
        # 2. Generate reconstructions

        X_rec = self.Gdd(z_gen).to(place)
        z_rec = self.Fed(X_gen).to(place)
        # 3. Cross-Discriminate XX
        # pdb.set_trace()
        Gloss_cycle_X = torch.mean(torch.abs(Xd.to(place)-X_rec)).to(place)
        
        # 4. Cross-Discriminate ZZ
        Gloss_cycle_z = torch.mean(torch.abs(zd-z_rec)).to(place)
        # Gloss_cycle_z = torch.mean(torch.sum(z_rec**2,dim=1)).to(place)

        Gloss = Gloss_ali + 10. * Gloss_cycle_X + 10* Gloss_cycle_z 

        Gloss.backward()
        
        self.oGdxz.step()
        zerograd(self.optzd)
        
        self.losses['Gloss_t'].append(Gloss.tolist()) 
        self.losses['Gloss_x'].append(Gloss_cycle_X.tolist())
        self.losses['Gloss_z'].append(Gloss_cycle_z.tolist())
        
    ####################
    ##### FILTERED #####
    ####################
    def alice_train_filtered_discriminator_adv_xz(self,Xf,zf):
#OK
        # Set-up training
        # pdb.set_trace()
        zerograd(self.optzf)
        
        self.Fef.eval(),self.Gdf.eval()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        #make sure that noises are at the same app.DEVICE as data
        wnx = wnx.to(Xf.app.DEVICE)
        wnz = wnz.to(zf.app.DEVICE)
        
        # pdb.set_trace()
        # 1. Concatenate inputs
        X_inp = zcat(Xf,wnx)
        z_inp = zcat(zf,wnz)

        # 2. Generate conditional samples
        X_gen = self.Gdf(z_inp)
        z_gen = self.Fef(X_inp)

        # 3. Cross-Discriminate XZ
        DXz,DzX,_,_ = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)
         
        # 4. Compute ALI discriminator loss
        Dloss_ali = -torch.mean(ln0c(DzX)+ln0c(1.0-DXz))
        # Dloss_ali = laploss1d(DzX,DXz)
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)
        
        # pdb.set_trace()
        # 2. Generate reconstructions
        X_rec = self.Gdf(z_gen)
        z_rec = self.Fef(X_gen) 
        # 3. Cross-Discriminate XX
        # pdb.set_trace()
        Dreal_X,Dfake_X = self.discriminate_filtered_xx(Xf,X_rec)
        
        # warning in the perpose of using the BCEloss the value should be greater than zero, 
        # so we apply a boolean indexing. BCEloss use log for calculation so the negative value
        # will lead to isses. Why do we have negative value? the LeakyReLU is not tranform negative value to zero
        # I recommanded using activation's function that values between 0 and 1, like sigmoid
        
        Dloss_ali_X = self.bce_loss(Dreal_X,o1l(Dreal_X))+\
                      self.bce_loss(Dfake_X,o1l(Dfake_X))
        # Dloss_ali_X = laploss1d(Dreal_X, Dreal_X) + laploss1d(Dfake_X,Dfake_X)
        Dloss_ali_X = Dloss_ali_X
            
        # 4. Cross-Discriminate ZZ
        # pdb.set_trace()
        Dreal_z,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)

        Dloss_ali_z = self.bce_loss(Dreal_z,o1l(Dreal_z))+\
                      self.bce_loss(Dfake_z,o0l(Dfake_z))
        # Dloss_ali_z = laploss1d(Dreal_z,o1l(Dreal_z)) + laploss1d(Dfake_z,o1l(Dfake_z))
        # Dloss_ali_z
        # torch.cuda.empty_cache()
        # GPUtil.showUtilization(all=True)
        # Total loss
        # pdb.set_trace()
        Dloss = Dloss_ali + 10*Dloss_ali_X + 100*Dloss_ali_z
        # print(Dloss.to("cuda"))
        # print(Dloss)
        # Dloss = Dloss_ali + Dloss_ali_X + Dloss_ali_z
        Dloss.backward(),self.oDfxz.step(),clipweights(self.Dfnets),zerograd(self.optzf)
        self.losses['Dloss'].append(Dloss.tolist())  
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())  
        self.losses['Dloss_ali_X'].append(Dloss_ali_X.tolist())  
        self.losses['Dloss_ali_z'].append(Dloss_ali_z.tolist())
        
    def alice_train_filtered_generator_adv_xz(self,Xf,zf):
#OK
        #Set-up training
        # pdb.set_trace()
        zerograd(self.optzf)
        self.Fef.train(),self.Gdf.train()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
         
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
         
        # 1. Concatenate inputsnex
        X_inp = zcat(Xf,wnx)
        z_inp = zcat(zf,wnz)
         
        # 2. Generate conditional samples
        X_gen = self.Gdf(z_inp)
        z_gen = self.Fef(X_inp) 
        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
         
        # 3. Cross-Discriminate XZ
        DXz,DzX,ftXz,ftzX = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)

        # 4. Compute ALI Generator loss
        Gloss_ali = torch.mean(-DXz+DzX)
        Gloss_ftm = 0.
        for rf,ff in zip(ftXz,ftzX):
            Gloss_ftm += torch.mean((rf-ff)**2)
         
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)
         
        # 2. Generate reconstructions
        X_rec = self.Gdf(z_gen)
        z_rec = self.Fef(X_gen) 
        # z_rec = latent_resampling(self.Fef(X_gen),nzf,wn1)
 
        # 3. Cross-Discriminate XX
        _,Dfake_X = self.discriminate_filtered_xx(Xf,X_rec)

        #ALICE
        # Gloss_ali_X = self.bce_loss(Dfake_X[0],o1l(Dfake_X[0]))
        # Gloss_ali_X = laploss1d(Dfake_X[0],o1l(Dfake_X[0]))
        #WGAN
        Gloss_ali_X = -(1.-torch.mean(Dfake_X[0]))

        Gloss_cycle_X = torch.mean(torch.abs(Xf-X_rec))
        # Gloss_cycle_X = laploss1d(Xf,X_rec)
        
        # 4. Cross-Discriminate ZZ
        _,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)
        #ALICE
        Gloss_ali_z = self.bce_loss(Dfake_z[0],o1l(Dfake_z[0]))
        #WGAN
        # Gloss_ali_z = -(1.-torch.mean(Dfake_z[0]))
        # Gloss_ali_z = laploss1d(Dfake_z[0],o1l(Dfake_z[0]))
        # Gloss_cycle_z = torch.mean(torch.sum(z_rec**2,dim=1))
        Gloss_cycle_z = torch.mean(torch.abs(zf-z_rec)**2)
        # Gloss_cycle_z = torch.mean(torch.abs(zf-z_rec))
        # Gloss_cycle_z = laploss1d(zf,z_rec)

        # Total Loss
        Gloss = (Gloss_ftm*0.7+Gloss_ali  *0.3)*(1.-0.7)/2.0 + \
            (Gloss_cycle_X*0.9+Gloss_ali_X*0.1)*(1.-0.1)/1.0 +\
            (Gloss_cycle_z*0.7+Gloss_ali_z*0.3)*(1.-0.7)/2.0
        Gloss.backward(),self.oGfxz.step(),zerograd(self.optzf)
         
        self.losses['Gloss'].append(Gloss.tolist()) 
        self.losses['Gloss_ftm'].append(Gloss_ali_X.tolist())
        self.losses['Gloss_ali_X'].append(Gloss_ali_X.tolist())
        self.losses['Gloss_ali_z'].append(Gloss_ali_z.tolist())
        self.losses['Gloss_cycle_X'].append(Gloss_cycle_X.tolist())
        self.losses['Gloss_cycle_z'].append(Gloss_cycle_z.tolist())

    ####################
    ##### HYBRID #####
    ####################
    def alice_train_hybrid_discriminator_adv_xz(self,Xd,zd,Xf,zf):
        # Set-up training
        zerograd(self.optzh)
        self.Gdd.eval(),self.Fef.eval(),self.Ghz.eval()
        self.DsrXd.train(),self.Dsrzd.train()
        
        # 0. Generate noise
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
         
        # 1. Concatenate inputs
        zf_inp = zcat(self.Fef(zcat(Xf,wnxf)),wnzf) # zf_inp = zcat(zf,wnzf)
         
        # 2. Generate conditional samples
        zd_gen = self.Ghz(zf_inp)
        
        # 3. Cross-Discriminate ZZ
        Dreal_z,Dfake_z = self.discriminate_hybrid_zz(zd,zd_gen)
        if self.style=='WGAN':
            Dloss_ali = -(torch.mean(Dreal_z)-torch.mean(Dfake_z))
        else:
            Dloss_ali = self.bce_loss(Dreal_z,o1l(Dreal_z))+\
                self.bce_loss(Dfake_z,o0l(Dfake_z))
        # Dloss_ali = laploss1d(Dreal_z,o1l(Dreal_z)) + laploss1d(Dfake_z,o0l(Dfake_z))
            
        # 1. Concatenate inputs
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
        zd_gen = zcat(zd_gen,wnzd)
        
        # 2. Generate reconstructions
        Xd_rec = self.Gdd(zd_gen)
        
        # 3. Cross-Discriminate XX
        Dreal_X,Dfake_X = self.discriminate_hybrid_xx(Xd,Xd_rec)
        if self.style=='WGAN':
            Dloss_ali_X = -(torch.mean(Dreal_X)-torch.mean(Dfake_X))
        else:
            Dloss_ali_X = self.bce_loss(Dreal_X,o1l(Dreal_X))+\
                self.bce_loss(Dfake_X,o0l(Dfake_X))
        # Dloss_ali_X = laploss1d(Dreal_X,o1l(Dreal_X)) + laploss1d(Dfake_X,o0l(Dfake_X))

        # Total loss
        Dloss = Dloss_ali + Dloss_ali_X 
        Dloss.backward(),self.oDhzdzf.step()
        if self.style=='WGAN':
            clipweights(self.Dhnets)
        zerograd(self.optzh)
        self.losses['Dloss'].append(Dloss.tolist())
        
    def alice_train_hybrid_generator_adv_xz(self,Xd,zd,Xf,zf):
        # Set-up training
        zerograd(self.optzh)
        self.Fef.train(),self.Gdd.train(),self.Ghz.train()
        self.DsrXd.train(),self.Dsrzd.train()
         
        # 0. Generate noise
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
         
        # 1. Concatenate inputs
        zf_inp = zcat(self.Fef(zcat(Xf,wnxf)),wnzf) # zf_inp = zcat(zf,wnzf)
         
        # 2. Generate conditional samples
        zd_gen = self.Ghz(zf_inp)
        
        # 3. Cross-Discriminate ZZ
        _,Dfake_z = self.discriminate_hybrid_zz(zd,zd_gen)
        if self.style=='WGAN':
            Gloss_ali = -torch.mean(Dfake_z)
        else:
            Gloss_ali = self.bce_loss(Dfake_z,o1l(Dfake_z))
        
        # 1. Concatenate inputs
        zd_gen = zcat(zd_gen,wnzd)
        
        # 2. Generate reconstructions
        Xd_rec = self.Gdd(zd_gen)
        
        # 3. Cross-Discriminate XX
        _,Dfake_X = self.discriminate_hybrid_xx(Xd,Xd_rec)
        if self.style=='WGAN':
            Gloss_ali_X = -torch.mean(Dfake_X)
        else:
            Gloss_ali_X = self.bce_loss(Dfake_X,o1l(Dfake_X))
        Xd_rec.retain_grad()
        Xf_rec = lowpass_biquad(Xd_rec,1./md['dtm'],md['cutoff']).to(app.DEVICE)
        #Gloss_cycle_Xd = torch.mean(torch.abs(Xd-Xd_rec))
        Gloss_cycle_Xf = torch.mean(torch.abs(Xf-Xf_rec))
        
        # Total Loss
        Gloss = Gloss_ali + Gloss_ali_X + 10.* Gloss_cycle_Xf 
        Gloss.backward(),self.oGhxz.step(),zerograd(self.optzh)
         
        self.losses['Gloss'].append(Gloss.tolist())
        
    # @profile
    def train_broadband(self):
        print('Training on broadband signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        error = {}
        
        start_time = time.time()
        for epoch in range(niter):
            for b,batch in enumerate(self.trn_loader):
                # Load batch
                # pdb.sclcet_trace()
                place = opt.dev if self.dp_mode else ngpu-1
                xd_data,_,zd_data,*other = batch
                Xd = Variable(xd_data).to(place) # BB-signal
                zd = Variable(zd_data).to(place)
                # Train G/D
                for _ in range(5):
                    self.alice_train_broadband_discriminator_explicit_xz(Xd,zd)
                    torch.cuda.empty_cache()
                
                for _ in range(1):
                    self.alice_train_broadband_generator_explicit_xz(Xd,zd)
                    torch.cuda.empty_cache()
            if epoch%10== 0:
                print("--- {} minutes ---".format((time.time() - start_time)/60))
                

            # GPUtil.showUtilization(all=True)
# TODO
#=======
#            for np in range(opt.dataloaders):
#                ths_trn = tload(opj(opt.dataroot,'ths_trn_ns{:>d}_nt{:>d}_ls{:>d}_nzf{:>d}_nzd{:>d}_{:>d}.pth'.format(nsy,
#                    opt.signalSize,opt.latentSize,opt.nzf,opt.nzd,np)))
#                # ths_tst = tload(opj(opt.dataroot,'ths_tst_{:>d}{:s}'.format(np,opt.dataset)))
#                # ths_vld = tload(opj(opt.dataroot,'ths_vld_{:>d}{:s}'.format(np,opt.dataset)))
#                trn_loader = AsynchronousLoader(trn_loader, app.DEVICE=app.DEVICE)
#                
#                for b,batch in enumerate(trn_loader):
#                    # Load batch
#                    # pdb.set_trace()
#                    xd_data,_,zd_data,_,_,_,_ = batch
#                    Xd = Variable(xd_data).to(ngpu-1,non_blocking=True) # BB-signal
#                    zd = Variable(zd_data).to(ngpu-1,non_blocking=True)
#                    # Train G/D
#                    for _ in range(5):
#                        self.alice_train_broadband_discriminator_explicit_xz(Xd,zd)
#                        torch.cuda.empty_cache()
#                    
#                    for _ in range(1):
#                        self.alice_train_broadband_generator_explicit_xz(Xd,zd)
#                        torch.cuda.empty_cache()
#            
#                GPUtil.showUtilization(all=True)
#
#>>>>>>> 63085b8... [BUGFIX]
            
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
            
            if save_checkpoint:
                if epoch%save_checkpoint==0:
                    print("\t|saving model at this checkpoint : ", epoch)
                    tsave({'epoch':epoch,'model_state_dict':self.Fed.state_dict(),
                           'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},
                           './network/{0}/Fed_{1}.pth'.format(outf[7:],epoch))
                    tsave({'epoch':epoch,'model_state_dict':self.Gdd.state_dict(),
                           'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},
                           './network/{0}/Gdd_{1}.pth'.format(outf[7:],epoch))    
                    tsave({'model_state_dict':self.Dszd.state_dict(),
                           'optimizer_state_dict':self.oDdxz.state_dict()},'./network/{0}/Dszd_bb_{1}.pth'.format(outf[7:],epoch))
                    tsave({'model_state_dict':self.DsXd.state_dict(),
                           'optimizer_state_dict':self.oDdxz.state_dict()},'./network/{0}/DsXd_bb_{1}.pth'.format(outf[7:],epoch))    
                    tsave({'model_state_dict':self.Ddxz.state_dict(),
                           'optimizer_state_dict':self.oDdxz.state_dict()},'./network/{0}/Ddxz_bb_{1}.pth'.format(outf[7:],epoch))
        
        plt.plot_loss_explicit(losses=self.losses["Dloss_t"], key="Dloss_t", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_x"], key="Gloss_x", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_z"], key="Gloss_z", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_z"], key="Gloss_z", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_t"], key="Gloss_t", outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["norm_grad"], key="norm_grad", outf=outf,niter=niter) 
        

        tsave({'epoch':niter,'model_state_dict':self.Fed.state_dict(),
            'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'./network/{0}/Fed.pth'.format(outf[7:]))
        tsave({'epoch':niter,'model_state_dict':self.Gdd.state_dict(),
            'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'./network/{0}/Gdd.pth'.format(outf[7:]))    
        tsave({'model_state_dict':self.Dszd.state_dict()},'./network/{0}/Dszd_bb.pth'.format(outf[7:]))
        tsave({'model_state_dict':self.DsXd.state_dict()},'./network/{0}/DsXd_bb.pth'.format(outf[7:]))    
        tsave({'model_state_dict':self.Ddxz.state_dict()},'./network/{0}/Ddxz_bb.pth'.format(outf[7:]))
         
    # @profile
    def train_filtered_explicit(self):
        print('Training on filtered signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        error = {}
        trn_loader = AsynchronousLoader(trn_loader, device=app.DEVICE)
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                # pdb.set_trace()
                _,xf_data,_,zf_data,_,_,_ = batch
                Xf = Variable(xf_data).to(app.DEVICE,non_blocking=True) # LF-signal
                zf = Variable(zf_data).to(app.DEVICE,non_blocking=True)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_filtered_discriminator_explicit_xz(Xf,zf)
                    torch.cuda.empty_cache()

                for _ in range(1):
                    self.alice_train_filtered_generator_explicit_xz(Xf,zf)
                    torch.cuda.empty_cache()

            # pdb.set_trace()
                err = self._error(self.Fef, self.Gdf,Xf,zf,app.DEVICE)
                a =  err.cpu().data.numpy().tolist()
                #error in percentage (%)
                if b in error:
                    error[b] = np.append(error[b], a)
                else:
                    error[b] = a

            # GPUtil.showUtilization(all=True)

            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
            if save_checkpoint:
                if epoch%save_checkpoint==0:
                    print("saving model at this checkpoint, talk few minutes ...")
                    tsave({'epoch':epoch,'model_state_dict':self.Fef.state_dict(),
                           'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                           './network/Fef_{}.pth'.format(epoch))
                    tsave({'epoch':epoch,'model_state_dict':self.Gdf.state_dict(),
                           'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                           './network/Gdf_{}.pth'.format(epoch))    
                    tsave({'model_state_dict':self.Dszf.state_dict(),
                           'optimizer_state_dict':self.oDfxz.state_dict()},'./network/Dszd_fl_{}.pth'.format(epoch))
                    tsave({'model_state_dict':self.DsXf.state_dict(),
                           'optimizer_state_dict':self.oDfxz.state_dict()},'./network/DsXd_fl_{}.pth'.format(epoch))    
                    tsave({'model_state_dict':self.Dfxz.state_dict(),
                           'optimizer_state_dict':self.oDfxz.state_dict()},'./network/Ddxz_fl_{}.pth'.format(epoch))
        
        plt.plot_error(error,outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},'./network/Fef.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},'./network/Gdf.pth')

    def train_filtered(self):
#OK
        print("[!] In function train_filtered ... ")
        globals().update(self.cv)
        globals().update(opt.__dict__)
        error = {}
        start_time = time.time()
        # path = './network/trained/nfz8/'
        
        for epoch in range(niter):
            # for np in range(opt.dataloaders):
            #     ths_trn = tload(opj(opt.dataroot,'ths_trn_ns{:>d}_nt{:>d}_ls{:>d}_nzf{:>d}_nzd{:>d}_{:>d}.pth'.format(nsy,
            #         opt.signalSize,opt.latentSize,opt.nzf,opt.nzd,np)))
            #            # ths_tst = tload(opj(opt.dataroot,'ths_tst_{:>d}{:s}'.format(np,opt.dataset)))
            #            # ths_vld = tload(opj(opt.dataroot,'ths_vld_{:>d}{:s}'.format(np,opt.dataset)))
            #     trn_loader = AsynchronousLoader(trn_loader, app.DEVICE=app.DEVICE)
            
            for b,batch in enumerate(self.trn_loader):
                # Load batch
                # pdb.set_trace()
                # print(b)
                # _,xf_data,_,zf_data,_,_,_ = batch
                
                # xf_data,_,zf_data,_,_,_,_ = batch # modifieddata (broadband)
                _,xf_data,_,zf_data = batch # modifieddata (broadband)
                Xf = Variable(xf_data).to(app.DEVICE,non_blocking=True) # LF-signal
                zf = Variable(zf_data).to(app.DEVICE,non_blocking=True)

                wnzd = torch.empty(*(opt.batchSize,32,64)).normal_(**app.RNDM_ARGS).to(app.DEVICE)
                # print("Xf and zf", Xf.shape, zf.shape)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_filtered_discriminator_adv_xz(Xf,zf)
                    torch.cuda.empty_cache()
                for _ in range(1):
                    self.alice_train_filtered_generator_adv_xz(Xf,zf)
                    torch.cuda.empty_cache()



                err = self._error(self.Fef, self.Gdf, Xf,zf,app.DEVICE)
                a =  err.cpu().data.numpy().tolist()
                #error in percentage (%)
                if b in error:
                    error[b] = np.append(error[b], a)
                else:
                    error[b] = a
            if epoch%10== 0:
                print("--- {} minutes ---".format((time.time() - start_time)/60))
                print(" saving discriminators ...")
            #     torch.save(f="./network/trained/nfz8/Dszf.pth",  obj=self.Dszf)
            #     torch.save(f="./network/trained/nfz8/Dsrzf.pth", obj=self.Dsrzf)
            #     torch.save(f="./network/trained/nfz8/DsXf.pth",  obj=self.DsXf)
            #     torch.save(f="./network/trained/nfz8/DsrXf.pth", obj=self.DsrXf)
            #     torch.save(f="./network/trained/nfz8/Dfxz.pth",  obj=self.Dfxz)
            # GPUtil.showUtilization(all=True)

            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
            if save_checkpoint:
                if epoch%save_checkpoint==0:
                    print("saving model at this checkpoint, less than 2 minutes ...")
                    tsave({'epoch':epoch,'model_state_dict':self.Fef.state_dict(),
                           'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                           './network/{0}/Fef_{1}.pth'.format(outf[7:],epoch))
                    tsave({'epoch':epoch,'model_state_dict':self.Gdf.state_dict(),
                           'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                           './network/{0}/Gdf_{1}.pth'.format(outf[7:],epoch))    
                    tsave({'model_state_dict':self.Dszf.state_dict(),
                           'optimizer_state_dict':self.oDfxz.state_dict()},'./network/{0}/Dszd_fl_{1}.pth'.format(outf[7:],epoch))
                    tsave({'model_state_dict':self.DsXf.state_dict(),
                           'optimizer_state_dict':self.oDfxz.state_dict()},'./network/{0}/DsXd_fl_{1}.pth'.format(outf[7:],epoch))    
                    tsave({'model_state_dict':self.Dfxz.state_dict(),
                           'optimizer_state_dict':self.oDfxz.state_dict()},'./network/{0}/Ddxz_fl_{1}.pth'.format(outf[7:],epoch))
        
        plt.plot_loss_explicit(losses=self.losses["Gloss"], key="Gloss", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Dloss"], key="Dloss", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_ftm"], key="Gloss_ftm", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_ali_X"], key="Gloss_ali_X", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_ali_z"], key="Gloss_ali_z", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_X"], key="Gloss_cycle_X", outf=outf,niter=niter)
        plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_z"], key="Gloss_cycle_z", outf=outf,niter=niter)

        tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'./network/Fef.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'./network/Gdf.pth')    
        tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dfnets],
            'optimizer_state_dict':self.oDfxz.state_dict(),'loss':self.losses},'./network/DsXz.pth')
    
    def train_hybrid(self):
        print('Training on filtered signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        error = {}
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                pdb.set_trace()
                xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
                Xd = Variable(xd_data).to(app.DEVICE) # BB-signal
                Xf = Variable(xf_data).to(app.DEVICE) # LF-signal
                zd = Variable(zd_data).to(app.DEVICE)
                zf = Variable(zf_data).to(app.DEVICE)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_hybrid_discriminator_adv_xz(Xd,zd,Xf,zf)
                for _ in range(1):
                    self.alice_train_hybrid_generator_adv_xz(Xd,zd,Xf,zf)

            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
        
        tsave({'epoch':niter,'model_state_dict':self.Ghz.state_dict(),
            'optimizer_state_dict':self.oGhxz.state_dict(),'loss':self.losses},'Ghz.pth')    
        tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dhnets],
            'optimizer_state_dict':self.oDhzdzf.state_dict(),'loss':self.losses},'DsXz.pth')

    # @profile
    def train(self):
        for t,a in self.strategy['tract'].items():
            if 'broadband' in t.lower() and a:
                self.train_broadband()
            if 'filtered' in t.lower() and a:                    
                self.train_filtered()
            if 'hybrid' in t.lower() and a:                    
                self.train_hybrid()

    # @profile            
    def generate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'broadband' in t and self.strategy['trplt']['broadband']:
            n = self.strategy['broadband']
            
            plt.plot_generate_classic('broadband',self.Fed,self.Gdd,app.DEVICE,vtm,\
                                      self.vld_loader,pfx="vld_set_bb",outf=outf)
            
            plt.plot_features('broadband',self.Fed,self.Gdd,nzd,app.DEVICE,vtm,vld_loader,pfx='set_bb',outf=outf)

        if 'filtered' in t and self.strategy['trplt']['filtered']:
            n = self.strategy['filtered']
            Fef = deepcopy(self.Fef)
            Gdf = deepcopy(self.Gdf)
            if None not in n:
                print("Loading models {} {}".format(n[0],n[1]))
                Fef.load_state_dict(tload(n[0])['model_state_dict'])
                Gdf.load_state_dict(tload(n[1])['model_state_dict'])
            
            plt.plot_generate_classic('filtered',Fef,Gdf,app.DEVICE,vtm,\
                                      self.vld_loader,pfx="vld_set_fl",outf=outf)
            
            # plt.plot_features('filtered',self.Fef,self.Gdf,nzf,app.DEVICE,vtm,self.vld_loader,pfx='set_fl',outf=outf)

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
            plt.plot_generate_hybrid(Fef,Gdd,Ghz,app.DEVICE,vtm,\
                                      trn_loader,pfx="trn_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gdd,Ghz,app.DEVICE,vtm,\
                                      tst_loader,pfx="tst_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gdd,Ghz,app.DEVICE,vtm,\
                                      vld_loader,pfx="vld_set_hb",outf=outf)

    def compare(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trcmp']['hybrid']:
            n = self.strategy['hybrid']
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            plt.plot_compare_ann2bb(self.Fef,self.Gdd,self.Ghz,app.DEVICE,vtm,\
                                    trn_loader,pfx="trn_set_ann2bb",outf=outf)
    def discriminate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trdis']['hybrid']:
            n = self.strategy['hybrid']
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
                self.Ddxz.load_state_dict(tload(n[3])['model_state_dict'])
                self.DsXd.load_state_dict(tload(n[4])['model_state_dict'])
                self.Dszd.load_state_dict(tload(n[5])['model_state_dict'])
                # import pdb
                #pdb.set_trace()
                DsXz = load_state_dict(tload(n[6])['model_state_dict'])
            # Set-up training
            self.Fef.eval(),self.Gdd.eval()
            self.DsXd.eval(),self.Dszd.eval(),self.Ddxz.eval()
            
            for epoch in range(niter):
                for b,batch in enumerate(trn_loader):
                    # Load batch
                    xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
                    Xd = Variable(xd_data).to(app.DEVICE) # BB-signal
                    Xf = Variable(xf_data).to(app.DEVICE) # LF-signal
                    zd = Variable(zd_data).to(app.DEVICE)
                    zf = Variable(zf_data).to(app.DEVICE)

    # @profile
    def _error(self,encoder, decoder, Xt, zt, dev):
        wnx,wnz,wn1 = noise_generator( Xt.shape,zt.shape,dev,app.RNDM_ARGS)
        X_inp = zcat(Xt,wnx.to(dev))
        ztr = encoder(X_inp).to(dev)
        # ztr = latent_resampling(Qec(X_inp),zt.shape[1],wn1)
        z_inp = zcat(ztr,wnz.to(dev))
        z_pre = zcat(zt,wnz.to(dev))
        Xr = decoder(z_inp)

        loss = torch.nn.MSELoss(reduction="mean")

        return loss(Xt,Xr)

    # def get_Dgrad_norm(self, Xd, Xdr, zd, zdr):
    #     self.Ddxz.train()
        
    #     batch_size = Xdr.shape[0]

    #     # Calculate interpolation
    #     alpha = torch.rand( size = (batch_size, 1, 1), requires_grad=True)
    #     alphaX = alpha.expand_as(Xd)
    #     alphaz = alpha.expand_as(zd)

    #     interpolated = torch.zeros_like(Xdr, requires_grad=True)

    #     if torch.cuda.is_available():
    #         alphaX = alphaX.cuda(ngpu-1)
    #         alphaz = alphaz.cuda(ngpu-1)

    #     Xhat = alphaX*Xd+(1-alphaX)*Xdr
    #     zhat = alphaz*zd+(1-alphaz)*zdr

    #     if torch.cuda.is_available():
    #         Xhat = Xhat.cuda()
    #         zhat = zhat.cuda()

    #     Xhat.requires_grad_(True)
    #     zhat.requires_grad_(True)

    #     prob_interpolated = self.Ddxz.critic(zcat(self.DsXd(Xhat),self.Dszd(zhat)))

    #     if torch.cuda.is_available():
    #         grad_outputs = torch.ones(prob_interpolated.size(), requires_grad=True).cuda(ngpu-1)
    #     else:
    #         grad_outputs = torch.ones(prob_interpolated.size(), requires_grad=True)

    #     prob_interpolated.requires_grad_(True)
    #     interpolated.requires_grad_(True)
    #     grad_outputs.requires_grad_(True)

    #     # Calculate gradients of probabilities with respect to examples
    #     gradXz = torch.autograd.grad(outputs=prob_interpolated,\
    #         inputs = (Xhat,zhat),\
    #         grad_outputs = grad_outputs,\
    #         create_graph = True,\
    #         retain_graph = True,
    #         only_inputs  = True)[0]

    #     # Gradients have shape (batch_size, num_channels, img_width, img_height),
    #     # so flatten to easily take norm per example in batch
    #     gradXz = gradXz.view(batch_size, -1)

    #     # Derivatives of the gradient close to 0 can cause problems because of
    #     # the square root, so manually calculate norm and add epsilon
    #     gradients_norm = torch.sqrt(torch.sum(gradXz ** 2, dim=1) + 1e-12)

    #     # Return gradient penalty
    #     return gradients_norm


