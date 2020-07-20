# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from profile_support import profile
from common_nn import *
from common_torch import * 
from dcgaae_model import Encoder, Decoder
from dcgaae_model import DCGAN_Dx, DCGAN_Dz
from dcgaae_model import DCGAN_DXX, DCGAN_DZZ, DCGAN_DXZ
from dcgaae_model import DenseEncoder
import plot_tools as plt
from generate_noise import latent_resampling, noise_generator
from generate_noise import lowpass_biquad
from database_sae import random_split 
from leave_p_out import k_folds
from common_setup import dataset2loader
from database_sae import thsTensorData

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
b1 = 0.5
b2 = 0.9999
nch_tot = 3
penalty_wgangp = 10.
nly = 5
#self.style='ALICE'#'WGAN'
acts={}
acts['ALICE'] = {'Fed' :[LeakyReLU(1.0,inplace=True) for t in range(nly)]+[LeakyReLU(1.0,inplace=True)],
                 'Gdd' :[ReLU(inplace=True) for t in range(nly-1)]+[Tanh()],
                 'Fef' :[LeakyReLU(1.0,inplace=True) for t in range(4)]+[LeakyReLU(1.0,inplace=True)],
                 'Gdf' :[ReLU(inplace=True) for t in range(4)]+[Tanh()],
                 'Ghz' :[ReLU(inplace=True) for t in range(2)]+[LeakyReLU(1.0,inplace=True)],
                 'Phz' :[ReLU(inplace=True) for t in range(1)]+[LeakyReLU(1.0,inplace=True)],
                 'Dsx' :[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 'Dsz' :[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 'Drx' :[LeakyReLU(1.0,inplace=True),Sigmoid()],
                 'Drz' :[LeakyReLU(1.0,inplace=True),Sigmoid()],
                 'Ddxz':[LeakyReLU(1.0,inplace=True),Sigmoid()],
                 'DhXd':[LeakyReLU(1.0,inplace=True) for t in range(2)]+[Sigmoid()]}

acts['WGAN']  = {'Fed' :[LeakyReLU(1.0,inplace=True) for t in range(nly)]+[LeakyReLU(1.0,inplace=True)],
                 'Gdd' :[ReLU(inplace=True) for t in range(nly-1)]+[Tanh()],
                 'Fef' :[LeakyReLU(1.0,inplace=True) for t in range(4)]+[LeakyReLU(1.0,inplace=True)],
                 'Gdf' :[ReLU(inplace=True) for t in range(4)]+[Tanh()],
                 'Ghz' :[ReLU(inplace=True) for t in range(2)]+[LeakyReLU(1.0,inplace=True)],
                 'Phz' :[ReLU(inplace=True) for t in range(1)]+[LeakyReLU(1.0,inplace=True)],
                 'Dsx' :[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 'Dsz' :[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 'Drx' :[LeakyReLU(1.0,inplace=True) for t in range(2)],
                 'Drz' :[LeakyReLU(1.0,inplace=True) for t in range(2)],
                 'Ddxz':[LeakyReLU(1.0,inplace=True) for t in range(2)],
                 'DhXd':[LeakyReLU(1.0,inplace=True) for t in range(3)]}
    
class trainer(object):
    '''Initialize neural network'''
    @profile
    def __init__(self,cv):
        super(trainer, self).__init__()
    
        self.cv = cv
        globals().update(cv)
        globals().update(opt.__dict__)
        self.strategy=strategy
        
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
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'broadband' in t:
            self.style='ALICE'
            act = acts[self.style]
            flagT = True
            n = self.strategy['broadband']
            print("Loading broadband generators")
            # Encoder broadband Fed
            self.Fed = Encoder(ngpu=ngpu,dev=device,nz=nzd,nzcl=0,nch=2*nch_tot,
                               ndf=ndf,szs=md['ntm'],nly=nly,ker=4,std=2,\
                               pad=0,dil=1,grp=1,dpc=0.0,act=act['Fed']).to(device)
            # Decoder broadband Gdd
            self.Gdd = Decoder(ngpu=ngpu,nz=2*nzd,nch=nch_tot,ndf=ndf//(2**(5-nly)),nly=nly,ker=4,std=2,pad=0,\
                               opd=0,dpc=0.0,act=act['Gdd']).to(device)
            if self.strategy['tract']['broadband']:
                if None in n:
                    self.FGd = [self.Fed,self.Gdd]
                    self.oGdxz = reset_net(self.FGd,func=set_weights,lr=glr,b1=b1,b2=b2,\
                            weight_decay=None)
                else:
                    print("Broadband generators: {0} - {1}".format(*n))
                    self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                    self.oGdxz = Adam(ittc(self.Fed.parameters(),self.Gdd.parameters()),
                                      lr=glr,betas=(b1,b2))#,weight_decay=None)
                self.optzd.append(self.oGdxz)
                self.Dszd = DCGAN_Dz(ngpu=ngpu,nz=nzd,ncl=512,n_extra_layers=1,dpc=0.25,
                                     bn=False,activation=act['Dsz']).to(device)
                self.DsXd = DCGAN_Dx(ngpu=ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
                                     n_extra_layers=0,dpc=0.25,activation=act['Dsx']).to(device)
                self.Ddxz = DCGAN_DXZ(ngpu=ngpu,nc=1024,n_extra_layers=2,dpc=0.25,
                                      activation=act['Ddxz']).to(device)    
                self.Ddnets.append(self.DsXd)  
                self.Ddnets.append(self.Dszd)
                self.Ddnets.append(self.Ddxz)
                self.oDdxz = reset_net(self.Ddnets,func=set_weights,lr=rlr,b1=b1,b2=b2)
                self.optzd.append(self.oDdxz)   
            else:
                if None not in n:
                    print("Broadband generators - NO TRAIN: {0} - {1}".format(*n))
                    self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                else:
                    flagT=False

        if 'filtered' in t:
            self.style='ALICE'
            act = acts[self.style]
            flagF = True
            n = self.strategy['filtered']
            print("Loading filtered generators")
            self.Fef = Encoder(ngpu=ngpu,dev=device,nz=nzf,nzcl=0,nch=2*nch_tot,
                               ndf=ndf,szs=md['ntm'],nly=5,ker=4,std=2,\
                               pad=0,dil=1,grp=1,dpc=0.0,\
                               act=act['Fef']).to(device)
            self.Gdf = Decoder(ngpu=ngpu,nz=2*nzf,nch=nch_tot,ndf=ndf,nly=5,ker=4,std=2,pad=0,\
                               opd=0,dpc=0.0,act=act['Gdf']).to(device)
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
                self.Dszf = DCGAN_Dz(ngpu=ngpu,nz=nzf,ncl=2*nzf,n_extra_layers=2,dpc=0.25,bn=False,
                                     activation=act['Dsz'],wf=True).to(device)
                self.DsXf = DCGAN_Dx(ngpu=ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
                                     n_extra_layers=0,dpc=0.25,activation=act['Dsx'],
                                     wf=True).to(device)
                self.Dfxz = DCGAN_DXZ(ngpu=ngpu,nc=512+2*nzf,n_extra_layers=2,dpc=0.25,
                                      activation=act['Ddxz'],wf=True).to(device)
                self.Dfnets.append(self.DsXf)
                self.Dfnets.append(self.Dszf)
                self.Dfnets.append(self.Dfxz)
                # recontruction
                self.Dsrzf = DCGAN_Dz(ngpu=ngpu,nz=2*nzf,ncl=2*nzf,n_extra_layers=1,dpc=0.25,
                                      bn=False,activation=act['Drz']).to(device)
                self.DsrXf = DCGAN_Dx(ngpu=ngpu,isize=256,nc=2*nch_tot,ncl=512,ndf=64,fpd=1,
                                      n_extra_layers=0,dpc=0.25,activation=act['Drx']).to(device)
                self.Dfnets.append(self.DsrXf)
                self.Dfnets.append(self.Dsrzf)
                self.oDfxz = reset_net(self.Dfnets,func=set_weights,lr=rlr,optim='rmsprop')
                self.optzf.append(self.oDfxz)
            else:
                if None not in n:
                    print("Filtered generators - no train: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdf.load_state_dict(tload(n[1])['model_state_dict'])    
                else:
                    flagF=False
        if 'hybrid' in t and flagF and flagT:
            self.style='WGAN'
            act = acts[self.style]
            n = self.strategy['hybrid']
            print("Loading hybrid generators")
            self.Ghz = Encoder(ngpu=ngpu,dev=device,nz=nzd,nzcl=0,nch=2*nzf,
                    ndf=nzf*4,szs=32,nly=3,ker=3,std=1,\
                            pad=1,dil=1,grp=1,dpc=0.0,bn=True,
                            act=act['Ghz']).to(device)
            if None not in n:
                print("Hybrid generators - no train: {0} - {1} - {2}".format(*n))
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            if self.strategy['tract']['hybrid'] or self.strategy['trdis']['hybrid']: 
                if self.style=='WGAN':
                    self.oGhxz = reset_net([self.Ghz],func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    self.oGhxz = reset_net([self.Ghz],func=set_weights,lr=glr,b1=b1,b2=b2)
                self.optzh.append(self.oGhxz)
                self.Dsrzd = DCGAN_DXZ(ngpu=ngpu,nc=2*nzd,n_extra_layers=2,dpc=0.25,
                                       activation=act['Ddxz'],wf=False).to(device)
                
                if self.style=='WGAN':
                    self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=False,\
                                     act=act['DhXd']).to(device)
                    self.Dhnets.append(self.Dsrzd)
                    self.Dhnets.append(self.DsrXd)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=True,\
                                     act=act['DhXd']).to(device)
                    self.Dhnets.append(self.Dsrzd)
                    self.Dhnets.append(self.DsrXd)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,b1=b1,b2=b2)
                self.optzh.append(self.oDhzdzf) 
        # Loss Criteria
        self.bce_loss = BCE(reduction='mean').to(device)
        self.losses = {'Dloss_t':[0],'Dloss_f':[0],'Dloss_t':[0],
                       'Gloss_x':[0],'Gloss_z':[0],'Gloss_t':[0],
                       'Gloss_xf':[0],'Gloss_xt':[0],'Gloss_f':[0],
                       'Gloss':[0],'Dloss':[0],'Gloss_ftm':[0],'Gloss_ali_X':[0],
                       'Gloss_ali_z':[0],'Gloss_cycle_X':[0],
                       'Gloss_cycle_z':[0],'Dloss_ali':[0],
                       'Dloss_ali_X':[0],'Dloss_ali_z':[0]}
        
    @profile
    def discriminate_broadband_xz(self,Xd,Xdr,zd,zdr):
        
        # Discriminate real
        zrc = zcat(self.DsXd(Xd),self.Dszd(zdr))
        DXz = self.Ddxz(zrc)
        
        # Discriminate fake
        zrc = zcat(self.DsXd(Xdr),self.Dszd(zd))
        DzX = self.Ddxz(zrc)
        
        return DXz,DzX

    @profile
    def discriminate_filtered_xz(self,Xf,Xfr,zf,zfr):
        
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

    @profile
    def discriminate_filtered_xx(self,Xf,Xfr):
        Dreal = self.DsrXf(zcat(Xf,Xf ))
        Dfake = self.DsrXf(zcat(Xf,Xfr))
        return Dreal,Dfake

    @profile
    def discriminate_filtered_zz(self,zf,zfr):
        Dreal = self.Dsrzf(zcat(zf,zf ))
        Dfake = self.Dsrzf(zcat(zf,zfr))
        return Dreal,Dfake

    @profile
    def discriminate_hybrid_xx(self,Xf,Xfr):
        Dreal = self.DsrXd(zcat(Xf,Xf ))
        Dfake = self.DsrXd(zcat(Xf,Xfr))
        return Dreal,Dfake

    @profile
    def discriminate_hybrid_zz(self,zf,zfr):
        Dreal = self.Dsrzd(zcat(zf,zf ))
        Dfake = self.Dsrzd(zcat(zf,zfr))
        return Dreal,Dfake

    ####################
    ##### CLASSIC  #####
    ####################
    @profile
    def alice_train_broadband_discriminator_explicit_xz(self,Xd,zd):
        
        # Set-up training
        zerograd(self.optzd)
        self.Fed.eval(),self.Gdd.eval()
        self.DsXd.train(),self.Dszd.train(),self.Ddxz.train()
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,device,rndm_args)
         
        # 1. Concatenate inputs
        X_inp = zcat(Xd,wnx)
        z_inp = zcat(zd,wnz)
        
        # 2. Generate conditional samples
        X_gen = self.Gdd(z_inp)
        z_gen = self.Fed(X_inp)
        # z_gen = latent_resampling(self.Fed(X_inp),nzd,wn1)
        
        # 3. Cross-Discriminate XZ
        Dxz,Dzx = self.discriminate_broadband_xz(Xd,X_gen,zd,z_gen)
        
        # 4. Compute ALI discriminator loss
        Dloss_ali = -torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        
        # Total loss
        Dloss = Dloss_ali
        Dloss.backward(),self.oDdxz.step(),zerograd(self.optzd)
        self.losses['Dloss_t'].append(Dloss.tolist())  
    
    @profile
    def alice_train_broadband_generator_explicit_xz(self,Xd,zd):
        
        # Set-up training
        zerograd(self.optzd)
        self.Fed.train(),self.Gdd.train()
        self.DsXd.train(),self.Dszd.train(),self.Ddxz.train()
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        
        # 1. Concatenate inputs
        X_inp = zcat(Xd,wnx)
        z_inp = zcat(zd,wnz)
        
        # 2. Generate conditional samples
        X_gen = self.Gdd(z_inp)
        z_gen = self.Fed(X_inp)
        # z_gen = latent_resampling(self.Fed(X_inp),nzd,wn1)
        
        # 3. Cross-Discriminate XZ
        Dxz,Dzx = self.discriminate_broadband_xz(Xd,X_gen,zd,z_gen)
        
        # 4. Compute ALI Generator loss
        Gloss_ali = torch.mean(-Dxz +Dzx)
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)
        
        # 2. Generate reconstructions
        X_rec = self.Gdd(z_gen)
        z_rec = self.Fed(X_gen)
        # z_rec = latent_resampling(self.Fed(X_gen),nzd,wn1)

        # 3. Cross-Discriminate XX
        Gloss_cycle_X = torch.mean(torch.abs(Xd-X_rec))   
        
        # 4. Cross-Discriminate ZZ
        Gloss_cycle_z = torch.mean(torch.abs(zd-z_rec))  

        # Total Loss
        Gloss = Gloss_ali + 10. * Gloss_cycle_X + 100. * Gloss_cycle_z
        Gloss.backward(),self.oGdxz.step(),zerograd(self.optzd)
        
        self.losses['Gloss_t'].append(Gloss.tolist()) 
        self.losses['Gloss_x'].append(Gloss_cycle_X.tolist())
        self.losses['Gloss_z'].append(Gloss_cycle_z.tolist())
    ####################
    ##### FILTERED #####
    ####################
    def alice_train_filtered_discriminator_adv_xz(self,Xf,zf):
        # Set-up training
        zerograd(self.optzf)
        self.Fef.eval(),self.Gdf.eval()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
         
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,device,rndm_args)
         
        # 1. Concatenate inputs
        X_inp = zcat(Xf,wnx)
        z_inp = zcat(zf,wnz)
         
        # 2. Generate conditional samples
        X_gen = self.Gdf(z_inp)
        z_gen = self.Fef(X_inp) 
        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
         
        # 3. Cross-Discriminate XZ
        DXz,DzX,_,_ = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)
         
        # 4. Compute ALI discriminator loss
        Dloss_ali = -torch.mean(ln0c(DzX)+ln0c(1.0-DXz))
        #Dloss_ali = -(torch.mean(DzX) - torch.mean(DXz))

        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)
         
        # 2. Generate reconstructions
        X_rec = self.Gdf(z_gen)
        z_rec = self.Fef(X_gen) 
        # z_rec = latent_resampling(self.Fef(X_gen),nzf,wn1)
        # 3. Cross-Discriminate XX
        Dreal_X,Dfake_X = self.discriminate_filtered_xx(Xf,X_rec)
        Dloss_ali_X = self.bce_loss(Dreal_X,o1l(Dreal_X))+\
            self.bce_loss(Dfake_X,o0l(Dfake_X))
            
                
        # 4. Cross-Discriminate ZZ
        Dreal_z,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)
        Dloss_ali_z = self.bce_loss(Dreal_z,o1l(Dreal_z))+\
            self.bce_loss(Dfake_z,o0l(Dfake_z))
        
        # Total loss
        Dloss = Dloss_ali + Dloss_ali_X + Dloss_ali_z
        Dloss.backward(),self.oDfxz.step(),clipweights(self.Dfnets),zerograd(self.optzf)
        self.losses['Dloss'].append(Dloss.tolist())  
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())  
        self.losses['Dloss_ali_X'].append(Dloss_ali_X.tolist())  
        self.losses['Dloss_ali_z'].append(Dloss_ali_z.tolist())
        
    def alice_train_filtered_generator_adv_xz(self,Xf,zf):
        # Set-up training
        zerograd(self.optzf)
        self.Fef.train(),self.Gdf.train()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
         
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,device,rndm_args)
         
        # 1. Concatenate inputs
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
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        
        # 1. Concatenate inputs
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
#         Gloss_ftmX = 0.
#         for rf,ff in zip(ftX_real,ftX_fake):
#             Gloss_ftmX += torch.mean((rf-ff)**2)
        
        # 4. Cross-Discriminate ZZ
        _,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)
        Gloss_ali_z = self.bce_loss(Dfake_z,o1l(Dfake_z))
        Gloss_cycle_z = torch.mean(torch.abs(zf-z_rec)**2)
#         Gloss_ftmz = 0.
#         for rf,ff in zip(ftz_real,ftz_fake):
#             Gloss_ftmz += torch.mean((rf-ff)**2)    
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
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
         
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
            
        # 1. Concatenate inputs
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
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
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
         
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
        Xf_rec = lowpass_biquad(Xd_rec,1./md['dtm'],md['cutoff']).to(device)
        #Gloss_cycle_Xd = torch.mean(torch.abs(Xd-Xd_rec))
        Gloss_cycle_Xf = torch.mean(torch.abs(Xf-Xf_rec))
        
        # Total Loss
        Gloss = Gloss_ali + Gloss_ali_X + 10.* Gloss_cycle_Xf 
        Gloss.backward(),self.oGhxz.step(),zerograd(self.optzh)
         
        self.losses['Gloss'].append(Gloss.tolist())
        
    @profile
    def train_broadband(self):
        print('Training on broadband signals') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        #nsy = trn_loader.dataset.dataset.inpZ[:,0,0].view(-1).data.numpy().size 
        #kk=0
        #import pdb
        #pdb.set_trace()
        #for train_idx, test_idx in k_folds(n_splits=5,subjects=nsy,frames=1):
        #    idx=range(train_idx.size)
        #    inpX=trn_loader.dataset.dataset.inpX[train_idx,:,:].clone().detach() 
        #    inpY=trn_loader.dataset.dataset.inpY[train_idx,:,:].clone().detach()
        #    inpZ=trn_loader.dataset.dataset.inpZ[train_idx,:,:].clone().detach()
        #    tar=trn_loader.dataset.dataset.tar[train_idx,:].clone().detach()
        #    trn_loader_p_out = thsTensorData(inpX,inpY,inpZ,tar,idx)

        #    _,trn_loader_p_out = random_split(trn_loader_p_out,[len(train_idx),0,0])
        #    params = {'batch_size':opt.batchSize,\
        #        'shuffle': True,'num_workers':int(opt.workers)}
        #    trn_loader_p_out,_,_ =  trn_loader_p_out
        #    trn_loader_p_out,_,_ = \
        #        dataset2loader(trn_loader_p_out,trn_loader_p_out,trn_loader_p_out,**params)   
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                xd_data,_,zd_data,_,_,_,_ = batch
                Xd = Variable(xd_data).to(device) # BB-signal
                zd = Variable(zd_data).to(device)
                # Train G/D
                for _ in range(5):
                    self.alice_train_broadband_discriminator_explicit_xz(Xd,zd)
                for _ in range(1):
                    self.alice_train_broadband_generator_explicit_xz(Xd,zd)
    
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
        plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_classic',outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fed.state_dict(),
            'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'Fed.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdd.state_dict(),
            'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'Gdd.pth')    
        tsave({'model_state_dict':self.Dszd.state_dict()},'Dszd_bb.pth')
        tsave({'model_state_dict':self.DsXd.state_dict()},'DsXd_bb.pth')    
        tsave({'model_state_dict':self.Ddxz.state_dict()},'Ddxz_bb.pth')
         
    @profile
    def train_filtered_explicit(self):
        print('Training on filtered signals') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                _,xf_data,_,zf_data,_,_,_ = batch
                Xf = Variable(xf_data).to(device) # LF-signal
                zf = Variable(zf_data).to(device)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_filtered_discriminator_explicit_xz(Xf,zf)
                for _ in range(1):
                    self.alice_train_filtered_generator_explicit_xz(Xf,zf)
    
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
        plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_filtered',outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},'Fef.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},'Gdf.pth')    
    @profile
    def train_filtered(self):
        print('Training on filtered signals') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                _,xf_data,_,zf_data,_,_,_ = batch
                Xf = Variable(xf_data).to(device) # LF-signal
                zf = Variable(zf_data).to(device)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_filtered_discriminator_adv_xz(Xf,zf)
                for _ in range(1):
                    self.alice_train_filtered_generator_adv_xz(Xf,zf)
    
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
        plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_filtered',outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'Fef.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'Gdf.pth')    
        tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dfnets],
            'optimizer_state_dict':self.oDfxz.state_dict(),'loss':self.losses},'DsXz.pth')
    
    @profile
    def train_hybrid(self):
        print('Training on filtered signals') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
                Xd = Variable(xd_data).to(device) # BB-signal
                Xf = Variable(xf_data).to(device) # LF-signal
                zd = Variable(zd_data).to(device)
                zf = Variable(zf_data).to(device)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_hybrid_discriminator_adv_xz(Xd,zd,Xf,zf)
                for _ in range(1):
                    self.alice_train_hybrid_generator_adv_xz(Xd,zd,Xf,zf)
    
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
        plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_hybrid',outf=outf)
        
        tsave({'epoch':niter,'model_state_dict':self.Ghz.state_dict(),
            'optimizer_state_dict':self.oGhxz.state_dict(),'loss':self.losses},'Ghz.pth')    
        tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dhnets],
            'optimizer_state_dict':self.oDhzdzf.state_dict(),'loss':self.losses},'DsXz.pth')

    @profile
    def train(self):
        for t,a in self.strategy['tract'].items():
            if 'broadband' in t.lower() and a:
                self.train_broadband()
            if 'filtered' in t.lower() and a:                    
                self.train_filtered()
            if 'hybrid' in t.lower() and a:                    
                self.train_hybrid()

    @profile            
    def generate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'broadband' in t and self.strategy['trplt']['broadband']:
            n = self.strategy['broadband']
            #plt.plot_generate_classic('broadband',Fed,Gdd,device,vtm,\
            #                          trn_loader,pfx="trn_set_bb",outf=outf)
            #plt.plot_generate_classic('broadband',Fed,Gdd,device,vtm,\
            #                          tst_loader,pfx="tst_set_bb",outf=outf)
         #   plt.plot_generate_classic('broadband',self.Fed,self.Gdd,device,vtm,\
         #                             vld_loader,pfx="vld_set_bb",outf=outf)
         #   plt.plot_gofs(tag=['broadband'],Fef=self.Fef,Gdf=self.Gdf,Fed=self.Fed,\
         #           Gdd=self.Gdd,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
         #           pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb'},\
         #           outf=outf)
            plt.plot_features('broadband',self.Fed,self.Gdd,nzd,device,vtm,vld_loader,pfx='set_bb',outf=outf)
        if 'filtered' in t and self.strategy['trplt']['filtered']:
            n = self.strategy['filtered']
            Fef = deepcopy(self.Fef)
            Gdf = deepcopy(self.Gdf)
            if None not in n:
                print("Loading models {} {}".format(n[0],n[1]))
                Fef.load_state_dict(tload(n[0])['model_state_dict'])
                Gdf.load_state_dict(tload(n[1])['model_state_dict'])
            #plt.plot_generate_classic('filtered',Fef,Gdf,device,vtm,\
            #                          trn_loader,pfx="trn_set_fl",outf=outf)
            #plt.plot_generate_classic('filtered',Fef,Gdf,device,vtm,\
            #                          tst_loader,pfx="tst_set_fl",outf=outf)
         ##   plt.plot_generate_classic('filtered',Fef,Gdf,device,vtm,\
         ##                             vld_loader,pfx="vld_set_fl",outf=outf)
         ##   plt.plot_gofs(tag=['filtered'],Fef=self.Fef,Gdf=self.Gdf,Fed=self.Fed,\
         ##           Gdd=self.Gdd,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
         ##           pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb'},\
         ##           outf=outf)
            plt.plot_features('filtered',self.Fef,self.Gdf,nzf,device,vtm,vld_loader,pfx='set_fl',outf=outf)

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

    @profile            
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
            plt.plot_compare_ann2bb(self.Fef,self.Gdd,self.Ghz,device,vtm,\
                                    trn_loader,pfx="trn_set_ann2bb",outf=outf)
    @profile            
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
                import pdb
                pdb.set_trace()
                DsXz = load_state_dict(tload(n[6])['model_state_dict'])
            # Set-up training
            self.Fef.eval(),self.Gdd.eval()
            self.DsXd.eval(),self.Dszd.eval(),self.Ddxz.eval()
            
            for epoch in range(niter):
                for b,batch in enumerate(trn_loader):
                    # Load batch
                    xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
                    Xd = Variable(xd_data).to(device) # BB-signal
                    Xf = Variable(xf_data).to(device) # LF-signal
                    zd = Variable(zd_data).to(device)
                    zf = Variable(zf_data).to(device)
                    wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
                    wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
                    X_inp = zcat(Xf,wnxf)
                    zfr = self.Fef(X_inp)
                    zdr = self.Ghz(zcat(zfr,wnzf))
                    z_inp = zcat(zdr,wnzd)
                    Xr = self.Gdd(z_inp)
                    import pdb
                    pdb.set_trace()
                    Dxz,Dzx = self.discriminate_broadband_xz(Xd,Xr,zd,zdr)
    

