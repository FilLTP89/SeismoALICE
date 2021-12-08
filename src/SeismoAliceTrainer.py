# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''
from copy import deepcopy
from profile_support import profile
from CommonNN import *
from CommonTorch import *
from SeismoAliceModels import Encoder, Decoder
from SeismoAliceModels import DCGAN_Dx, DCGAN_Dz
from SeismoAliceModels import DCGAN_DXX, DCGAN_DZZ, DCGAN_DXZ
import PlotTools as plt
from GenerateNoise import latent_resampling, GetNoise
from GenerateNoise import lowpass_biquad
from LoadDatabase import random_split 
from leave_p_out import k_folds
from CommonSetup import dataset2loader
from LoadDatabase import thsTensorData
import json

rndm_args = {'mean': 0, 'std': 1}

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
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

class trainer(object):

    @profile
    def __init__(self,cv):
        super(trainer, self).__init__()

        self.cv = cv
        globals().update(cv)
        globals().update(opt.__dict__)

        # read configuration for networks
        with open(config,'r') as jsonfile:
            configs = jsonfile.read()
        configs = json.loads(configs)

        # Load encoder
        self.F = Encoder(d_x=md['ntm'],d_z=[nzxy,nzyy])
        self.F.get(net_type='branched',config=configs["encoders"]["F"]).to(device)

        # Load decoder
        self.F = Encoder(d_x=md['ntm'],d_z=[nzxy,nzyy]).get(net_type='branched',config=configs["encoders"]["F"]).to(device)

        # self.Gxx = Module()
        # self.Gyy = Module()

        # self.DsXd = Module()
        # self.Dszd = Module()
        # self.DsXf = Module()
        # self.Dszf = Module()

        # self.Ddnets = []
        # self.Dfnets = []
        # self.Dhnets = []
        # self.optzd  = []
        # self.optzf  = []
        # self.optzh  = []
        # self.oGdxz=None
        # self.oGfxz=None
        # self.oGhxz=None
        # flagT=False
        # flagF=False

        # self.style='ALICE'
        # act = acts[self.style]
        # flagT = True
        # n = self.strategy['broadband']
        # print("Loading broadband generators")
        # # Encoder broadband Fed
        
        # # Decoder broadband Gdd
        # self.Gdd = Decoder(ngpu=ngpu,nz=2*nzd,nch=nch_tot,
        #                    ndf=ndf//(2**(5-nlayers['Gdd'])),
        #                    nly=nlayers['Gdd'],ker=kernels['Gdd'],
        #                    std=strides['Gdd'],pad=padding['Gdd'],\
        #                    opd=outpads['Gdd'],dpc=0.0,act=act['Gdd']).to(device)

        # if None in n:
        #     self.FGd = [self.Fed,self.Gdd]
        #     self.oGdxz = reset_net(self.FGd,func=set_weights,lr=glr,b1=b1,b2=b2,\
        #             weight_decay=None)
        # else:
        #     print("Broadband generators: {0} - {1}".format(*n))
        #     self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
        #     self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
        #     self.oGdxz = Adam(ittc(self.Fed.parameters(),self.Gdd.parameters()),
        #                       lr=glr,betas=(b1,b2))#,weight_decay=None)
        # self.optzd.append(self.oGdxz)
        # self.Dszd = DCGAN_Dz(ngpu=ngpu,nz=nzd,ncl=512,n_extra_layers=1,dpc=0.25,
        #                      bn=False,activation=act['Dsz']).to(device)
        # self.DsXd = DCGAN_Dx(ngpu=ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
        #                      n_extra_layers=0,dpc=0.25,activation=act['Dsx']).to(device)
        # self.Ddxz = DCGAN_DXZ(ngpu=ngpu,nc=1024,n_extra_layers=2,dpc=0.25,
        #                       activation=act['Ddxz']).to(device)    
        # self.Ddnets.append(self.DsXd)  
        # self.Ddnets.append(self.Dszd)
        # self.Ddnets.append(self.Ddxz)
        # self.oDdxz = reset_net(self.Ddnets,func=set_weights,lr=rlr,b1=b1,b2=b2)
        # self.optzd.append(self.oDdxz)   
        # # else:
        # #     if None not in n:
        # #         print("Broadband generators - NO TRAIN: {0} - {1}".format(*n))
        # #         self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
        # #         self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
        # #     else:
        # #         flagT=False

        # # Loss Criteria
        # self.bce_loss = BCE(reduction='mean').to(device)
        # self.losses = {'Dloss_t':[0],'Dloss_f':[0],'Dloss_t':[0],
        #                'Gloss_x':[0],'Gloss_z':[0],'Gloss_t':[0],
        #                'Gloss_xf':[0],'Gloss_xt':[0],'Gloss_f':[0],
        #                'Gloss':[0],'Dloss':[0],'Gloss_ftm':[0],'Gloss_ali_X':[0],
        #                'Gloss_ali_z':[0],'Gloss_cycle_X':[0],
        #                'Gloss_cycle_z':[0],'Dloss_ali':[0],
        #                'Dloss_ali_X':[0],'Dloss_ali_z':[0]}

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

    def alice_train_discriminator(self,Xf,zf):
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
        
    def alice_train_generator(self,Xf,zf):
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

    @profile
    def train(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                y,x,*others = batch
                y = y.to(device) # BB-signal
                x = x.to(device) # LF-signal
                import pdb
                pdb.set_trace()
                zxy = tFT(y.shape[0],nzxy*latentSize).resize_(nsy,nzxy,latentSize).normal_(**rndm_args)
                zyy = tFT(y.shape[0],nzyy*latentSize).resize_(nsy,nzyy,latentSize).normal_(**rndm_args)
            #     # Train G/D
            #     for _ in range(5):
            #         self.alice_train_discriminator(y,zyy,x,zxy)
            #     for _ in range(1):
            #         self.alice_train_generator(Xf,zf)

            # str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            # str = 'epoch: {:>d} --- '.format(epoch)
            # str = str + ' | '.join(str1)
            # print(str)
        #     if epoch%save_checkpoint==0:
        #         tsave({'epoch':epoch,'model_state_dict':self.Fef.state_dict(),
        #                'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
        #                'Fef_{}.pth'.format(epoch))
        #         tsave({'epoch':epoch,'model_state_dict':self.Gdf.state_dict(),
        #                'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
        #                'Gdf_{}.pth'.format(epoch))    
        #         tsave({'model_state_dict':self.Dszf.state_dict(),
        #                'optimizer_state_dict':self.oDfxz.state_dict()},'Dszd_fl_{}.pth'.format(epoch))
        #         tsave({'model_state_dict':self.DsXf.state_dict(),
        #                'optimizer_state_dict':self.oDfxz.state_dict()},'DsXd_fl_{}.pth'.format(epoch))    
        #         tsave({'model_state_dict':self.Dfxz.state_dict(),
        #                'optimizer_state_dict':self.oDfxz.state_dict()},'Ddxz_fl_{}.pth'.format(epoch))
        # plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_filtered',outf=outf)
        # tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
        #     'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'Fef.pth')
        # tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
        #     'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'Gdf.pth')    
        # tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dfnets],
        #     'optimizer_state_dict':self.oDfxz.state_dict(),'loss':self.losses},'DsXz.pth')

    # @profile            
    # def generate(self):
    #     globals().update(self.cv)
    #     globals().update(opt.__dict__)

        # t = [y.lower() for y in list(self.strategy.keys())]
        # if 'broadband' in t and self.strategy['trplt']['broadband']:
        #     n = self.strategy['broadband']
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
        # plt.plot_features('broadband',self.Fed,self.Gdd,nzd,device,vtm,vld_loader,pfx='set_bb',outf=outf)