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
                 'Fhz' :[ReLU(inplace=True) for t in range(2)]+[LeakyReLU(1.0,inplace=True)],
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
                 'Fhz' :[ReLU(inplace=True) for t in range(2)]+[LeakyReLU(1.0,inplace=True)],
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
            if self.strategy['tract']['broadband'] or self.strategy['trdis']['hybrid']:
                if None in n:
                    self.FGd = [self.Fed,self.Gdd]
                    self.oGdxz = reset_net(self.FGd,func=set_weights,lr=glr,b1=b1,b2=b2,
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
                    print("Filtered generators - NO TRAIN: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdf.load_state_dict(tload(n[1])['model_state_dict'])    
                else:
                    flagF=False
        if 'hybrid' in t and flagF and flagT:
            self.style='WGAN'
            act = acts[self.style]
            n = self.strategy['hybrid']
            print("Loading hybrid generators")
            #self.Fhz = Encoder(ngpu=ngpu,dev=device,nz=nzf,nzcl=0,nch=2*nch_tot,
            #                   ndf=ndf,szs=md['ntm'],nly=5,ker=4,std=2,\
            #                   pad=0,dil=1,grp=1,dpc=0.0,\
            #                   act=act['Fef']).to(device)
            self.Fhz = Decoder(ngpu=ngpu,nz=2*nzd,nch=nzf,ndf=ndf,nly=3,ker=3,std=1,pad=1,\
                               opd=0,dpc=0.0,act=act['Fhz']).to(device)
            self.Ghz = Encoder(ngpu=ngpu,dev=device,nz=nzd,nzcl=0,nch=2*nzf,
                               ndf=nzf*4,szs=32,nly=3,ker=3,std=1,\
                               pad=1,dil=1,grp=1,dpc=0.0,bn=True,
                               act=act['Ghz']).to(device)
            if None not in n:
                print("Hybrid generators - NO TRAIN: {0} - {1}".format(*n))
                self.Fhz.load_state_dict(tload(n[0])['model_state_dict'])
                self.Ghz.load_state_dict(tload(n[1])['model_state_dict'])
            if self.strategy['tract']['hybrid'] or self.strategy['trdis']['hybrid']: 
                if self.style=='WGAN':
                    print("Generator Optimizer for WGAN")
                    self.oGhxz = reset_net([self.Fhz,self.Ghz],func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    self.oGhxz = reset_net([self.Fhz,self.Ghz],func=set_weights,lr=glr,b1=b1,b2=b2)
                self.optzh.append(self.oGhxz)
                self.Dsrzd = DCGAN_DXZ(ngpu=ngpu,nc=2*nzd,n_extra_layers=2,dpc=0.25,
                                       activation=act['Ddxz'],wf=False).to(device)
                self.Dsrzf = DCGAN_DXZ(ngpu=ngpu,nc=2*nzf,n_extra_layers=2,dpc=0.25,
                                       activation=act['Ddxz'],wf=False).to(device)
                self.Dhnets.append(self.Dsrzd)
                self.Dhnets.append(self.Dsrzf)
                if self.style=='WGAN':
                    print("Discriminator Optimizer for WGAN")
                    self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=False,\
                                     act=act['DhXd']).to(device)
                    self.DsrXf = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=False,\
                                     act=act['DhXd']).to(device)
                    self.Dhnets.append(self.DsrXd)
                    self.Dhnets.append(self.DsrXf)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=5,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=True,\
                                     act=act['DhXd']).to(device)
                    self.DsrXf = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=5,ker=3,std=2,\
                                     pad=1,dil=1,grp=1,dpc=0.25,bn=True,\
                                     act=act['DhXd']).to(device)
                    self.Dhnets.append(self.DsrXd)
                    self.Dhnets.append(self.DsrXf)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,b1=b1,b2=b2)
                self.optzh.append(self.oDhzdzf) 
        # Loss Criteria
        self.bce_loss = BCE(reduction='mean').to(device)
        self.losses = {} 
        self.losses[r"$l_y^D$"]=[0]
        self.losses[r"$l_x^D$"]=[0]
        self.losses[r"$l_{z^'}^D$"]=[0]
        self.losses[r"$l_z^D$"]=[0]
        self.losses[r"$l^D$"]=[0]
        self.losses[r"$l_y^G$"]=[0]
        self.losses[r"$l_x^G$"]=[0]
        self.losses[r"$l_{z^'}^G$"]=[0]
        self.losses[r"$l_z^G$"]=[0]
        self.losses[r"$l_{R1-y}$"]=[0]
        self.losses[r"$l_{R1-x}$"]=[0]
        self.losses[r"$l^G$"]=[0]
        

    @profile
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
    def discriminate_hybrid_xd(self,Xf,Xfr):
        Dreal = self.DsrXd(zcat(Xf,Xf ))
        Dfake = self.DsrXd(zcat(Xf,Xfr))
        return Dreal,Dfake
    
    @profile
    def discriminate_hybrid_xf(self,Xf,Xfr):
        Dreal = self.DsrXf(zcat(Xf,Xf ))
        Dfake = self.DsrXf(zcat(Xf,Xfr))
        return Dreal,Dfake


    @profile
    def discriminate_hybrid_zd(self,zf,zfr):
        Dreal = self.Dsrzd(zcat(zf,zf ))
        Dfake = self.Dsrzd(zcat(zf,zfr))
        return Dreal,Dfake
    
    @profile
    def discriminate_hybrid_zf(self,zf,zfr):
        Dreal = self.Dsrzf(zcat(zf,zf ))
        Dfake = self.Dsrzf(zcat(zf,zfr))
        return Dreal,Dfake

    ####################
    ##### HYBRID #####
    ####################
    def alice_train_hybrid_discriminator_adv_xz(self,Xd,zd,Xf,zf):
        # Set-up training
        zerograd(self.optzh)
        self.Gdd.eval(),self.Fef.eval()
        self.Ghz.eval(),self.Fhz.eval()
        self.DsrXd.train(),self.DsrXf.train()
        self.Dsrzd.train(),self.Dsrzf.train() 
        # 0. Generate noise
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
         
        # 1. Concatenate inputs
        zf_inp = zcat(self.Fef(zcat(Xf,wnxf)),wnzf) # zf_inp = zcat(zf,wnzf)
        zd_inp = zcat(self.Fed(zcat(Xd,wnxd)),wnzd) # zf_inp = zcat(zf,wnzf)
         
        # 2. Generate conditional samples
        zd_gen = self.Ghz(zf_inp)
        zf_gen = self.Fhz(zd_inp)
        # 3. Cross-Discriminate ZZ
        Dreal_zd,Dfake_zd = self.discriminate_hybrid_zd(zd,zd_gen)
        Dreal_zf,Dfake_zf = self.discriminate_hybrid_zf(zf,zf_gen)
        if self.style=='WGAN':
            Dloss_zd = -(torch.mean(Dreal_zd)-torch.mean(Dfake_zd))
            Dloss_zf = -(torch.mean(Dreal_zf)-torch.mean(Dfake_zf))
        else:
            Dloss_zd = self.bce_loss(Dreal_zd,o1l(Dreal_zd))+\
                self.bce_loss(Dfake_zd,o0l(Dfake_zd))
            Dloss_zf = self.bce_loss(Dreal_zf,o1l(Dreal_zf))+\
                self.bce_loss(Dfake_zf,o0l(Dfake_zf))
            
        # 1. Concatenate inputs
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        _,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        zd_gen = zcat(zd_gen,wnzd)
        zf_gen = zcat(zf_gen,wnzf)
        
        # 2. Generate reconstructions
        Xd_rec = self.Gdd(zd_gen)
        Xf_rec = self.Gdf(zf_gen)
        
        # 3. Cross-Discriminate XX
        Dreal_Xd,Dfake_Xd = self.discriminate_hybrid_xd(Xd,Xd_rec)
        Dreal_Xf,Dfake_Xf = self.discriminate_hybrid_xf(Xf,Xf_rec)
        if self.style=='WGAN':
            Dloss_ali_Xd = -(torch.mean(Dreal_Xd)-torch.mean(Dfake_Xd))
            Dloss_ali_Xf = -(torch.mean(Dreal_Xf)-torch.mean(Dfake_Xf))
        else:
            Dloss_ali_Xd = self.bce_loss(Dreal_Xd,o1l(Dreal_Xd))+\
                self.bce_loss(Dfake_Xd,o0l(Dfake_Xd))
            Dloss_ali_Xf = self.bce_loss(Dreal_Xf,o1l(Dreal_Xf))+\
                self.bce_loss(Dfake_Xf,o0l(Dfake_Xf))

        # Total loss
        Dloss = Dloss_zd + Dloss_zf + Dloss_ali_Xd + Dloss_ali_Xf 
        Dloss.backward(),self.oDhzdzf.step()
        if self.style=='WGAN':
            clipweights(self.Dhnets)
        zerograd(self.optzh)
        self.losses[r"$l_y^D$"].append(Dloss_ali_Xd.tolist())
        self.losses[r"$l_x^D$"].append(Dloss_ali_Xf.tolist())
        self.losses[r"$l_{z^'}^D$"].append(Dloss_zd.tolist())
        self.losses[r"$l_z^D$"].append(Dloss_zf.tolist())
        self.losses[r"$l^D$"].append(Dloss.tolist())

        
    def alice_train_hybrid_generator_adv_xz(self,Xd,zd,Xf,zf):
        # Set-up training
        zerograd(self.optzh)
        self.Gdd.train(),self.Fef.train()
        self.Ghz.train(),self.Fhz.train()
        self.DsrXd.train(),self.DsrXf.train()
        self.Dsrzd.train(),self.Dsrzf.train()
         
        # 0. Generate noise
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
         
        # 1. Concatenate inputs
        zf_inp = zcat(self.Fef(zcat(Xf,wnxf)),wnzf)
        zd_inp = zcat(self.Fed(zcat(Xd,wnxd)),wnzd)
        
        # 2. Generate conditional samples
        zd_gen = self.Ghz(zf_inp)
        zf_gen = self.Fhz(zd_inp)

        # 3. Cross-Discriminate ZZ
        _,Dfake_zd = self.discriminate_hybrid_zd(zd,zd_gen)
        _,Dfake_zf = self.discriminate_hybrid_zf(zf,zf_gen)
        if self.style=='WGAN':
            Gloss_ali_zf = -torch.mean(Dfake_zf)
            Gloss_ali_zd = -torch.mean(Dfake_zd)
        else:
            Gloss_ali_zf = self.bce_loss(Dfake_zf,o1l(Dfake_zf))
            Gloss_ali_zd = self.bce_loss(Dfake_zd,o1l(Dfake_zd))
        
        # 1. Concatenate inputs
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        _,wnzb,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        _,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        _,wnzc,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        zd_gen = zcat(zd_gen,wnzd)
        zc_gen = zcat(zf_gen,wnzc) 
        zf_gen = zcat(zf_gen,wnzf)
        # 2. Generate reconstructions
        Xd_rec = self.Gdd(zd_gen)
        Xf_rec = self.Gdf(zf_gen)
        zc_gen = self.Ghz(zc_gen)
        zc_gen = zcat(zc_gen,wnzb)
        Xd_fil = self.Gdd(zc_gen)
        
        # 3. Cross-Discriminate XX
        _,Dfake_Xd = self.discriminate_hybrid_xd(Xd,Xd_rec)
        _,Dfake_Xf = self.discriminate_hybrid_xf(Xf,Xf_rec)
        if self.style=='WGAN':
            Gloss_ali_Xd = -torch.mean(Dfake_Xd)
            Gloss_ali_Xf = -torch.mean(Dfake_Xf)
        else:
            Gloss_ali_Xd = self.bce_loss(Dfake_Xd,o1l(Dfake_Xd))
            Gloss_ali_Xf = self.bce_loss(Dfake_Xf,o1l(Dfake_Xf))
                
        ## Xd_rec.retain_grad()
        ## Xf_rec = lowpass_biquad(Xd_rec,1./md['dtm'],md['cutoff']).to(device)
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,device,rndm_args)
        zd_gen = zcat(self.Fed(zcat(Xf,wnxf)),wnzd) # zf_inp = zcat(zf,wnzf)
        zf_gen = zcat(self.Fhz(zd_gen),wnzf)
        Xf_rec = self.Gdf(zf_gen)
        Gloss_cycle_Xd = torch.mean(torch.abs(Xd-Xd_fil))
        Gloss_cycle_Xf = torch.mean(torch.abs(Xf-Xf_rec))
        
        # Total Loss
        Gloss = Gloss_ali_zf + Gloss_ali_zd + Gloss_ali_Xd + Gloss_ali_Xf + Gloss_cycle_Xf + 0.1*Gloss_cycle_Xd
        Gloss.backward(),self.oGhxz.step(),zerograd(self.optzh)


        self.losses[r"$l_y^G$"].append(Gloss_ali_Xd.tolist())
        self.losses[r"$l_x^G$"].append(Gloss_ali_Xf.tolist())
        self.losses[r"$l_{z^'}^G$"].append(Gloss_ali_zd.tolist())
        self.losses[r"$l_z^G$"].append(Gloss_ali_zf.tolist())
        self.losses[r"$l_{R1-y}$"].append(Gloss_cycle_Xd.tolist())
        self.losses[r"$l_{R1-x}$"].append(Gloss_cycle_Xf.tolist())
        self.losses[r"$l^G$"].append(Gloss.tolist())
        
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
        #plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_hybrid',outf=outf)
        
        tsave({'epoch':niter,'model_state_dict':self.Ghz.state_dict(),
            'optimizer_state_dict':self.oGhxz.state_dict(),'loss':self.losses},'Ghz.pth')    
        tsave({'epoch':niter,'model_state_dict':self.Fhz.state_dict(),
            'optimizer_state_dict':self.oGhxz.state_dict(),'loss':self.losses},'Fhz.pth')    
        tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dhnets],
            'optimizer_state_dict':self.oDhzdzf.state_dict(),'loss':self.losses},'DsXz.pth')

    @profile
    def train(self):
        for t,a in self.strategy['tract'].items():
            if 'hybrid' in t.lower() and a:                    
                self.train_hybrid()

    @profile            
    def generate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]

        if 'hybrid' in t and self.strategy['trplt']['hybrid']:
            n = self.strategy['hybrid']
            if None not in n:
                loss = tload(n[0])['loss']
            else:
                loss = self.losses
            #plt.plot_loss_dict(loss,nb=len(trn_loader),title='loss_hybrid',outf=outf)
            #plt.plot_generate_hybrid_new(self.Fef,self.Gdf,self.Fed,self.Gdd,self.Fhz,self.Ghz,device,vtm,\
            #                             trn_loader,pfx="trn_set_hb",outf=outf)
            #plt.plot_generate_hybrid_new(self.Fef,self.Gdf,self.Fed,self.Gdd,self.Fhz,self.Ghz,device,vtm,\
            #                             tst_loader,pfx="tst_set_hb",outf=outf)
            #plt.plot_generate_hybrid_new(self.Fef,self.Gdf,self.Fed,self.Gdd,self.Fhz,self.Ghz,device,vtm,\
            #                             vld_loader,pfx="vld_set_hb",outf=outf)
            plt.plot_gofs(tag=['hybrid'],Fef=self.Fef,Gdf=self.Gdf,Fed=self.Fed,\
                    Gdd=self.Gdd,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
                    pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb','ann2bb':'set_ann2bb_rec'},\
                    outf=outf)

    @profile            
    def compare(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trcmp']['hybrid']:
            n = self.strategy['hybrid']
            plt.plot_compare_ann2bb(self.Fef,self.Gdf,self.Fed,self.Gdd,self.Fhz,self.Ghz,device,vtm,\
                                    trn_loader,pfx="trn_ann2bb",outf=outf)
            plt.plot_compare_ann2bb(self.Fef,self.Gdf,self.Fed,self.Gdd,self.Fhz,self.Ghz,device,vtm,\
                                    tst_loader,pfx="tst_ann2bb",outf=outf)
            plt.plot_compare_ann2bb(self.Fef,self.Gdf,self.Fed,self.Gdd,self.Fhz,self.Ghz,device,vtm,\
                                    vld_loader,pfx="vld_ann2bb",outf=outf)
            #plt.plot_gofs(tag=['ann2bb'],Fef=self.Fef,Gdf=self.Gdf,Fed=self.Fed,\
            #              Gdd=self.Gdd,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
            #              pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb','ann2bb':'set_ann2bb_rec'},\
            #              outf=outf)
    @profile            
    def discriminate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trdis']['hybrid']:
            n = self.strategy['hybrid']
            self.Ddxz.load_state_dict(tload(n[2])['model_state_dict'])
            self.DsXd.load_state_dict(tload(n[3])['model_state_dict'])
            self.Dszd.load_state_dict(tload(n[4])['model_state_dict'])
            Dh = tload(n[5])['model_state_dict']
            self.DsrXd.load_state_dict(Dh[2])
            print("dpc initial : {}".format(self.Ddxz._modules['ann'][2]._modules['ann'][1].dpc))
            self.Ddxz._modules['ann'][2]._modules['ann'][1] = Dpout(dpc=0.0)
            print("dpc initial : {}".format(self.Ddxz._modules['ann'][2]._modules['ann'][1].dpc))
            plt.seismo_test(tag=['ann2bb'],Fef=self.Fef,Gdd=self.Gdd,Ghz=self.Ghz,Fed=self.Fed,\
                            Ddxz=self.Ddxz,DsXd=self.DsXd,Dszd=self.Dszd,DsrXd=self.DsrXd,\
                            dev=device,trn_set=trn_loader,\
                            pfx={'hybrid':'set_hb','ann2bb':'set_ann2bb'},\
                            outf=outf)

    

