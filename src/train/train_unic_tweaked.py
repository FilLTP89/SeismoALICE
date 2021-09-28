# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from copy import deepcopy
# from profile_support import profile
from common_nn import *
from common_torch import * 
import plot_tools as plt
from generate_noise import latent_resampling, noise_generator
from generate_noise import lowpass_biquad
from database_sae import random_split 
from leave_p_out import k_folds
from ex_common_setup import dataset2loader
from database_sae import thsTensorData
import json
# import pprint as pp
import pdb
from pytorch_summary import summary
from conv_factory import *
import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP
import time

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

# coder en dure dans le programme 
b1 = 0.5
b2 = 0.9999
nch_tot = 3
penalty_wgangp = 10.
nly = 5
#self.style='ALICE'#'WGAN'
"""
    global variable for this python file. 
    acts    [dictionnary]   :content ALICE and WGAN definitions.
                            Those latter represent common activation functions used in 
                            the whole prgramm for training The keyword :
                            + Fyx is the Encoder of Broadband signal, herein broadband 
                            signal is named Xd
                            + Gy is the Decoder(generator) Broadband, herein encoded 
                            signal form broad band is named zd 
                            + Gx is the Decoder(generator) of the filtred signal, herein 
                            filtred signal is named Xf
                            + Ghz is the Decoder(generator) of the hybrid signal
                            + Dsx is associated to DCGAN_Dx, and Dsxf and Dsxd(function)
                            + Dsz is associated to DCGAN_Dz(function)
                            + Drx is associated to DCGAN_Dx and DsrXf (functions)
                            + Drz is associated to DCGAN_Dx and Dsrzf(function)
                            + Ddxz is associated to DCGAN_DXZ, and Ddxf, Dfxz and Dsrzd
                            + [!] DhXd is for the Encoder and is associated to DsrXd for the 
                            hybride signal

    nlayers [dictionnary]   :contents the number of layers, parameter used for Encode and Decond on
                            a Conv1D functions.

    kernels [dictionnary]   :contents kernel_size parameter.

    strides [dictionnary]   :this parameter define the number of time stake that each 
                            convolutional window kernel will not see when this 
                            latter moves on the nt times points. 

    padding [dictionnary]   :contents the padding parameter relevant for the Conv1D  functions
    outpads [dictionnary]   :not used actually.
"""
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
        #import pdb
        # pdb.set_trace()
        #print(opt.config['decoder'])
        # passing the content of file ./strategy_bb_*.txt
        self.strategy = strategy
        self.ngpu     = ngpu

        nzd = opt.nzd
        ndf = opt.ndf
        
        # the follwings variable are the instance for the object Module from 
        # the package pytorch: torch.nn.modulese. /.
        # the names of variable are maped to the description aforementioned
        self.F_  = Module() # Single Encoder
        self.Gy  = Module() # Decoder for recorded signals
        self.Gx  = Module() # Decoder for synthetic signals
        self.Dy  = Module() # Discriminator for recorded signals
        self.Dx  = Module() # Discriminator for synthetic signals
        self.Dz  = Module() # Discriminator for latent space
        self.Dyz = Module() # Discriminator for (y,zd)
        self.Dxz = Module() # Discriminator for (x,zf)
        # Dyz(Dy(y),Dz(z))  /  Dxz(Dx(x),Dz(z))
        self.Dzz = Module() # Discriminator for latent space reconstrution
        self.Dxx = Module() # Discriminator for synthetic signal reconstrution
        self.Dyy = Module() # Discriminator for recorded signal reconstruction

        self.Dnets = []
        self.optz  = []
        self.oGyx  = None
        self.dp_mode = True

        # glr = 0.0001
        # rlr = 0.0001
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        cpus  =  int(os.environ.get('SLURM_NPROCS'))
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

        if 'unique' in t:
            self.style='ALICE'
            # act = acts[self.style]
            n = self.strategy['unique']
            # pdb.set_trace()
            self.F_  = net.Encoder(opt.config['F'], opt)
            self.Gy  = net.Decoder(opt.config['Gy'], opt)
            self.Gx  = net.Decoder(opt.config['Gx'], opt)
            
            if  self.strategy['tract']['unique']:
                if None in n:        
                    self.FGf  = [self.F_,self.Gy,self.Gx]
                    self.oGyx = reset_net(self.FGf,
                        func=set_weights,lr=glr,b1=b1,b2=b2,
                        weight_decay=0.00001)
                else:   
                    print("Unique encoder/Multi decoder: {0} - {1}".format(*n))
                    self.F_.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    self.Gx.load_state_dict(tload(n[2])['model_state_dict'])
                    self.oGyx = Adam(ittc(self.F_.parameters(),
                        self.Gy.parameters(),
                        self.Gx.parameters()),
                        lr=glr,betas=(b1,b2),
                        weight_decay=0.00001)

                self.optz.append(self.oGyx)

                ## Discriminators
                #pdb.set_trace()

                self.Dy    =  net.DCGAN_Dx(opt.config['Dy'],opt)
                self.Dx    =  net.DCGAN_Dx(opt.config['Dx'],opt)
                self.Dzb   =  net.DCGAN_Dz(opt.config['Dzb'],opt)
                self.Dzf   =  net.DCGAN_Dz(opt.config['Dzf'],opt)
                self.Dyz   =  net.DCGAN_DXZ(opt.config['Dyz'],opt)
                self.Dxz   =  net.DCGAN_DXZ(opt.config['Dxz'],opt)
                self.Dzzb  =  net.DCGAN_Dz(opt.config['Dzzb'], opt)
                self.Dzzf  =  net.DCGAN_Dz(opt.config['Dzzf'], opt)
                self.Dxx   =  net.DCGAN_Dx(opt.config['Dxx'], opt)
                self.Dyy   =  net.DCGAN_Dx(opt.config['Dyy'], opt)

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

                self.oDyxz = reset_net(self.Dnets,
                    func=set_weights,lr=rlr,
                    optim='Adam',b1=b1,b2=b2,
                    weight_decay=0.00001)
                self.optz.append(self.oDyxz)

            else:
                if None not in n:
                    self.F_.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
                    self.Gx.load_state_dict(tload(n[2])['model_state_dict'])  
                else:
                    flagF=False

        self.bce_loss = BCE(reduction='mean').to(ngpu-1)
        self.losses = {
            'Gloss':[0],'Dloss':[0],
                       'Gloss_ali_z':[0],
                       'Gloss_cycle_xy':[0],
                       'Gloss_ali_xy':[0],
                       # 'Gloss_cycle_z':[0],
                       'Dloss_ali':[0],
                       'Dloss_ali_x':[0],
                       'Dloss_ali_y':[0],
                       # 'Dloss_ali_xy':[0],
                       # 'Dloss_ali_z':[0]
                       }
        # pdb.set_trace()
    # @profile
    def discriminate_xz(self,x,xr,z,zr):
        # Discriminate real
        # pdb.set_trace()
        ftz = self.Dzf(zr)
        ftx = self.Dx(x)

        zrc = zcat(ftx[0],ftz[0])
        ftr = ftz[1]+ftx[1]
        
        ftxz = self.Dxz(zrc)
        Dxz  = ftxz[0]
        ftr += ftxz[1]
        
        # Discriminate fake
        ftz = self.Dzf(z)
        ftx = self.Dx(xr)

        zrc = zcat(ftx[0],ftz[0])
        ftf = ftz[1]+ftx[1]

        ftzx = self.Dxz(zrc)
        Dzx  = ftzx[0]
        ftf += ftzx[1]

        return Dxz,Dzx,ftr,ftf

    def discriminate_yz(self,x,xr,z,zr):
        # Discriminate real
        # pdb.set_trace()
        ftz = self.Dzb(zr)
        ftx = self.Dy(x)

        # zrc = zcat(ftx[0],ftz[0]).reshape(-1,1)
        zrc = zcat(ftx[0],ftz[0])
        ftr = ftz[1]+ftx[1]
        
        ftxz = self.Dyz(zrc)
        Dxz  = ftxz[0]
        ftr += ftxz[1]
        
        # Discriminate fake
        ftz = self.Dzb(z)
        ftx = self.Dy(xr)
        
        # zrc = zcat(ftx[0],ftz[0]).reshape(-1,1)
        zrc = zcat(ftx[0],ftz[0])
        ftf = ftz[1]+ftx[1]

        ftzx = self.Dyz(zrc)
        Dzx  = ftzx[0]
        ftf += ftzx[1]

        return Dxz,Dzx,ftr,ftf
    
    ''' Methode that discriminate real and fake hybrid signal type'''
    def discriminate_hybrid_xz(self,Xd,Xdr,zd,zdr):
        
        # Discriminate real
        ftz = self.Dzf(zdr)
        ftX = self.Dy(Xd)
        zrc = zcat(ftX[0],ftz[0])
        ftr = ftz[1]+ftX[1]
        ftXz = self.Dhzdzf(zrc)
        DXz  = ftXz[0]
        ftr += ftXz[1]
        
        # Discriminate fake
        ftz = self.Dzf(zd)
        ftX = self.Dy(Xdr)
        zrc = zcat(ftX[0],ftz[0])
        ftf = ftz[1]+ftX[1]
        ftzX = self.Dhzdzf(zrc)
        DzX  = ftzX[0]
        ftf += ftzX[1]
        
        return DXz,DzX,ftr,ftf

    # @profile
    def discriminate_xx(self,x,xr):
        # pdb.set_trace()
        Dreal = self.Dxx(zcat(x,x ))
        Dfake = self.Dxx(zcat(x,xr))
        return Dreal,Dfake

    def discriminate_yy(self,x,xr):
        # pdb.set_trace()
        Dreal = self.Dyy(zcat(x,x ))
        Dfake = self.Dyy(zcat(x,xr))
        return Dreal,Dfake

    # @profile
    def discriminate_zzb(self,z,zr):
        # pdb.set_trace()
        Dreal = self.Dzzb(zcat(z,z ))
        Dfake = self.Dzzb(zcat(z,zr))
        return Dreal,Dfake

    def discriminate_zzf(self,z,zr):
        # pdb.set_trace()
        Dreal = self.Dzzf(zcat(z,z ))
        Dfake = self.Dzzf(zcat(z,zr))
        return Dreal,Dfake

    # # @profile
    # def discriminate_hybrid_xx(self,Xf,Xfr):
    #     Dreal = self.DsrXd(zcat(Xf,Xf ))
    #     Dfake = self.DsrXd(zcat(Xf,Xfr))
    #     return Dreal,Dfake

    # # @profile
    # def discriminate_hybrid_zz(self,zf,zfr):
    #     Dreal = self.Dsrzd(zcat(zf,zf ))
    #     Dfake = self.Dsrzd(zcat(zf,zfr))
    #     return Dreal,Dfake

    def alice_train_discriminator_adv(self,y,zd,x,zf):
        # Set-up training        
        zerograd(self.optz)
        self.F_.eval()  , self.Gx.eval()  , self.Gy.eval()
        self.Dy.train() , self.Dx.train() , self.Dz.train()
        self.Dyz.train(), self.Dxz.train(), self.Dzzb.train()
        self.Dyy.train(), self.Dxx.train(), self.Dzzf.train()
        
        # 0. Generate noise
        wnx,wnzf,wn1 = noise_generator(x.shape,zf.shape,device,rndm_args)
        wny,wnzd,wn1 = noise_generator(y.shape,zd.shape,device,rndm_args)

        wny = wny.to(y.device)
        wnx = wnx.to(x.device)
        wnzd = wnzd.to(zd.device)
        wnzf = wnzf.to(zf.device)

        # 1. Concatenate inputs
        y_inp  = zcat(y,wny)
        x_inp  = zcat(x,wnx)
        zd_inp = zcat(zd,wnzd)
        zf_inp = zcat(zf,wnzf)

        # pdb.set_trace()

        # 2. Generate conditional samples
        y_gen = self.Gy(zd_inp)
        x_gen = self.Gx(zf_inp)

        zd_gen,_  = self.F_(y_inp)
        _, zf_gen = self.F_(x_inp)

        # zd_gen = self.F_(y_inp)
        # zf_gen = self.F_(x_inp)

        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
        # print("\t||X_gen : ", X_gen.shape,"\tz_gen : ",z_gen.shape)
        # 3. Cross-Discriminate XZ
        # pdb.set_trace()
        Dyz,Dzy,_,_ = self.discriminate_yz(y,y_gen,zd,zd_gen)
        Dxz,Dzx,_,_ = self.discriminate_xz(x,x_gen,zf,zf_gen)
         
        # 4. Compute ALI discriminator loss
        # Dloss_ali = -torch.mean(ln0c(Dzy)+ln0c(1.0-Dyz))-torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        Dloss_ali_y = self.bce_loss(Dzy,o1l(Dzy)) + self.bce_loss(Dyz,o0l(Dyz))
        Dloss_ali_x = self.bce_loss(Dzx,o1l(Dzx)) + self.bce_loss(Dxz,o0l(Dxz)) #-(torch.mean(DzX) - torch.mean(DXz))
        Dloss_ali = Dloss_ali_x+Dloss_ali_y

        wnx,wnzf,wn1 = noise_generator(x.shape,zf.shape,device,rndm_args)
        wny,wnzd,wn1 = noise_generator(y.shape,zd.shape,device,rndm_args)

        wny = wny.to(y.device)
        wnx = wnx.to(x.device)
        wnzd = wnzd.to(zd.device)
        wnzf = wnzf.to(zf.device)

        # 1. Concatenate inputs
        y_gen  = zcat(y_gen,wny)
        x_gen  = zcat(x_gen,wnx)
        zd_gen = zcat(zd_gen,wnzd)
        zf_gen = zcat(zf_gen,wnzf)
        
        # pdb.set_trace()
        # 2. Generate reconstructions
        y_rec  = self.Gy(zd_gen)
        x_rec  = self.Gx(zf_gen)

        zd_rec, _ = self.F_(y_gen)
        _, zf_rec = self.F_(x_gen)

        # zd_rec = self.F_(y_gen)
        # zf_rec = self.F_(x_gen)
        # z_rec = latent_resampling(self.Fef(X_gen),nzf,wn1)

        # 3. Cross-Discriminate XX
        Dreal_y,Dfake_y = self.discriminate_yy(y,y_rec)
        Dreal_x,Dfake_x = self.discriminate_xx(x,x_rec)

        Dloss_rec_y = self.bce_loss(Dreal_y,o1l(Dreal_y))+self.bce_loss(Dfake_y,o0l(Dfake_y))
        Dloss_rec_x = self.bce_loss(Dreal_x,o1l(Dreal_x))+self.bce_loss(Dfake_x,o0l(Dfake_x))
        Dloss_rec = Dloss_rec_x+Dloss_rec_y
        #warning in the perpose of us ing the BCEloss the value should be greater than zero, 
        # so we apply a boolean indexing. BCEloss use log for calculation so the negative value
        # will lead to isses. Why do we have negative value? the LeakyReLU is not tranform negative value to zero
        # I recommanded using activation's function that values between 0 and 1, like sigmoid
        # pdb.set_trace()
        # Dloss_ali_xy = self.bce_loss(Dreal_y,o1l(Dreal_y))+self.bce_loss(Dfake_y,o1l(Dfake_y))+\
        #     self.bce_loss(Dreal_x,o1l(Dreal_x))+self.bce_loss(Dfake_x,o1l(Dfake_x))
            
            
        # 4. Cross-Discriminate ZZ
        # pdb.set_trace()
        Dreal_zd,Dfake_zd = self.discriminate_zzb(zd,zd_rec)
        Dreal_zf,Dfake_zf = self.discriminate_zzf(zf,zf_rec)
        # Dreal_z, Dfake_z  = self.discriminate_zz(zf,zd)
        
        Dloss_rec_z = self.bce_loss(Dreal_zd,o1l(Dreal_zd))+self.bce_loss(Dfake_zd,o0l(Dfake_zd))+\
            self.bce_loss(Dreal_zf,o1l(Dreal_zf))+self.bce_loss(Dfake_zf,o0l(Dfake_zf))
            # self.bce_loss(Dreal_z,o1l(Dreal_z))+self.bce_loss(Dfake_z,o0l(Dfake_z))
       
        # Total loss
        Dloss = Dloss_ali + Dloss_rec_x + Dloss_rec_z + Dloss_rec
        Dloss.backward(),self.oDyxz.step() #,clipweights(self.Dnets),
        zerograd(self.optz)
        self.losses['Dloss'].append(Dloss.tolist())  
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())  
        self.losses['Dloss_ali_x'].append(Dloss_ali_x.tolist())
        self.losses['Dloss_ali_y'].append(Dloss_ali_y.tolist())
        # self.losses['Dloss_ali_xy'].append(Dloss_ali_xy.tolist())  
        # self.losses['Dloss_ali_z'].append(Dloss_ali_z.tolist())

    def alice_train_generator_adv(self,y,zd,x,zf):
        # Set-up training
        zerograd(self.optz)
        self.F_.train() , self.Gx.train() , self.Gy.train()
        self.Dy.train() , self.Dx.train() , self.Dz.train()
        self.Dyz.train(), self.Dxz.train() ,self.Dzzb.train()
        self.Dyy.train(), self.Dxx.train(), self.Dzzf.train()


         
        # 0. Generate noise
        wny,wnzd,wn1 = noise_generator(y.shape,zd.shape,device,rndm_args)
        wnx,wnzf,wn1 = noise_generator(x.shape,zf.shape,device,rndm_args)
         
        # 1. Concatenate inputs
        y_inp  = zcat(y,wny)
        x_inp  = zcat(x,wnx)
        zd_inp = zcat(zd,wnzd)
        zf_inp = zcat(zf,wnzf)

        # pdb.set_trace()
         
        # 2. Generate conditional samples
        y_gen = self.Gy(zd_inp) #(100,64,64)->(100,3,4096)
        x_gen = self.Gx(zf_inp)
        # zd_gen = self.F_(y_inp)[:,:zd.shape[1]] #(100,64,64) -> (100,32,64)
       # pdb.set_trace()
        zd_gen, _ = self.F_(y_inp)
        _, zf_gen = self.F_(x_inp)

        # zf_gen = zcat(zf_gen,wnzf) # = dim zd_gen
        # zd_gen = self.F_(y_inp)
        # zf_gen = self.F_(x_inp)
        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
         
        # 3. Cross-Discriminate XZ
        # pdb.set_trace()
        Dyz,Dzy,ftyz,ftzy = self.discriminate_yz(y,y_gen,zd,zd_gen)
        Dxz,Dzx,ftxz,ftzx = self.discriminate_xz(x,x_gen,zf,zf_gen)

        # 4. Compute ALI Generator loss
        Gloss_ali = torch.mean(-Dyz+Dzy)+torch.mean(-Dxz+Dzx)
        Gloss_ftm = 0.0
        for rf,ff in zip(ftxz,ftzx):
            Gloss_ftm += torch.mean((rf-ff)**2)
        for rf,ff in zip(ftyz,ftzy):
            Gloss_ftm += torch.mean((rf-ff)**2)

        # 0. Generate noise
        wny,wnzd,wn1 = noise_generator(y.shape,zd.shape,device,rndm_args)
        wnx,wnzf,wn1 = noise_generator(x.shape,zf.shape,device,rndm_args)
        
        # 1. Concatenate inputs
        y_gen  = zcat(y_gen,wny)
        x_gen  = zcat(x_gen,wnx) 
        zd_gen = zcat(zd_gen,wnzd) #(100,64,64)
        zf_gen = zcat(zf_gen,wnzf) #(100,32,64)

        # 2. Generate reconstructions
        y_rec = self.Gy(zd_gen)
        x_rec = self.Gx(zf_gen)
        # zd_rec = self.F_(y_gen)[:,:zd.shape[1]]

        zd_rec, _ = self.F_(y_gen)
        _, zf_rec = self.F_(x_gen)

        # zd_rec = self.F_(y_gen) 
        # zf_rec = self.F_(x_gen)
        # z_rec = latent_resampling(self.Fef(X_gen),nzf,wn1)
 
        # 3. Cross-Discriminate XX
        _,Dfake_y = self.discriminate_yy(y,y_rec)
        _,Dfake_x = self.discriminate_xx(x,x_rec)
        Gloss_ali_xy = self.bce_loss(Dfake_y,o1l(Dfake_y))+self.bce_loss(Dfake_x,o1l(Dfake_x))
        Gloss_cycle_xy = torch.mean(torch.abs(y-y_rec))+torch.mean(torch.abs(x-x_rec))
        # Gloss_ftmX = 0.
#         for rf,ff in zip(ftX_real,ftX_fake):
#             Gloss_ftmX += torch.mean((rf-ff)**2)
        
        # 4. Cross-Discriminate ZZ
        # pdb.set_trace()
        _,Dfake_zd = self.discriminate_zzb(zd,zd_rec)
        # pdb.set_trace()
        _,Dfake_zf = self.discriminate_zzf(zf,zf_rec)
        Gloss_ali_z = self.bce_loss(Dfake_zd,o1l(Dfake_zd))+self.bce_loss(Dfake_zf,o1l(Dfake_zf))
        # Gloss_cycle_z = torch.mean(torch.abs(zd-zd_rec))+torch.mean(torch.abs(zf-zf_rec))
#         Gloss_ftmz = 0.
#         for rf,ff in zip(ftz_real,ftz_fake):
#             Gloss_ftmz += torch.mean((rf-ff)**2)    
        # Total Loss
        Gloss = Gloss_ftm+Gloss_ali_xy+Gloss_cycle_xy+Gloss_ali_z
        # Gloss = (Gloss_ftm*0.7)*(1.-0.7)/2.0 + \
        #     (Gloss_cycle_xy*0.9+Gloss_ali_xy*0.1)*(1.-0.1)/1.0 +\
        #     (Gloss_ali_z*0.7)*(1.-0.7)/2.0
        Gloss.backward(),self.oGyx.step(),zerograd(self.optz)
         
        self.losses['Gloss'].append(Gloss.tolist()) 
        self.losses['Gloss_cycle_xy'].append(Gloss_cycle_xy.tolist())
        self.losses['Gloss_ali_z'].append(Gloss_ali_z.tolist())
        self.losses['Gloss_ali_xy'].append(Gloss_ali_xy.tolist())
        # self.losses['Gloss_cycle_X'].append(Gloss_cycle_X.tolist())
        # self.losses['Gloss_cycle_z'].append(Gloss_cycle_z.tolist())

        
    # @profile
    def train_unique(self):
        print('Training on both recorded and synthetic signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        error = {}
        place = opt.dev if self.dp_mode else ngpu-1
        start_time = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(niter):
                # for np in range(opt.dataloaders):
                    # ths_trn = tload(opj(opt.dataroot,'ths_trn_ns{:>d}_nt{:>d}_ls{:>d}_nzf{:>d}_nzd{:>d}_{:>d}.pth'.format(nsy,
                    #     opt.signalSize,opt.latentSize,opt.nzf,opt.nzd,np)))
                    # # ths_tst = tload(opj(opt.dataroot,'ths_tst_{:>d}{:s}'.format(np,opt.dataset)))
                    # # ths_vld = tload(opj(opt.dataroot,'ths_vld_{:>d}{:s}'.format(np,opt.dataset)))
                    # trn_loader = AsynchronousLoader(trn_loader, device=device)
                    # Load batch
                for b,batch in enumerate(trn_loader):
                    y,x,zd,zf,_,_,_ = batch
                    y   = y.to(place) # recorded signal
                    x   = x.to(place) # synthetic signal
                    zd  = zd.to(place) # recorded signal latent space
                    zf  = zf.to(place) # synthetic signal latent space
                # Train G/D
               
                for _ in range(5):
                    self.alice_train_discriminator_adv(y,zd,x,zf)
                    # torch.cuda.empty_cache()
                
                for _ in range(1):
                    self.alice_train_generator_adv(y,zd,x,zf)
                    # torch.cuda.empty_cache()
                
                # GPUtil.showUtilization(all=True)
                if epoch%10== 0:
                    print("--- {} minutes ---".format((time.time() - start_time)/60))

                str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
                str = 'epoch: {:>d} --- '.format(epoch)
                str = str + ' | '.join(str1)
                print(str)

                
                if save_checkpoint:
                    if epoch%save_checkpoint==0:
                        print("\t|saving model at this checkpoint : ", epoch)
                        tsave({'epoch':epoch,'model_state_dict':self.F_.state_dict(),
                               'optimizer_state_dict':self.oGyx.state_dict(),'loss':self.losses,},
                               './network/{0}/Fyx_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Gy.state_dict(),
                               'optimizer_state_dict':self.oGyx.state_dict(),'loss':self.losses,},
                               './network/{0}/Gy_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Gx.state_dict(),
                               'optimizer_state_dict':self.oGyx.state_dict(),'loss':self.losses,},
                               './network/{0}/Gx_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dy.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dy_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dx.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dx_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dz.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dz_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dxz.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dxz_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dzz.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dzz_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dxx.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dxx_{1}.pth'.format(outf[7:],epoch))
                        tsave({'epoch':epoch,'model_state_dict':self.Dyy.state_dict(),
                               'optimizer_state_dict':self.oDyxz.state_dict(),'loss':self.losses,},
                               './network/{0}/Dyy_{1}.pth'.format(outf[7:],epoch))

                        # tsave({'model_state_dict':self.Dzd.state_dict(),
                        #        'optimizer_state_dict':self.oDdxz.state_dict()},'./network/Dzd_bb_{}.pth'.format(epoch))
                        # tsave({'model_state_dict':self.Dy.state_dict(),
                        #        'optimizer_state_dict':self.oDdxz.state_dict()},'./network/Dy_bb_{}.pth'.format(epoch))    
                        # tsave({'model_state_dict':self.Ddxz.state_dict(),
                        #        'optimizer_state_dict':self.oDdxz.state_dict()},'./network/Ddxz_bb_{}.pth'.format(epoch))
            # pdb.set_trace()
            # plt.plot_loss_dict(nb=niter,losses=self.losses,title='loss_classic',outf=outf)
            plt.plot_loss_explicit(losses=self.losses["Dloss_ali"],     key="Dloss_ali",     outf=outf,niter=niter)
            plt.plot_loss_explicit(losses=self.losses["Dloss_ali_x"],   key="Dloss_ali_x",   outf=outf,niter=niter)
            plt.plot_loss_explicit(losses=self.losses["Dloss_ali_y"],   key="Dloss_ali_y",   outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Dloss_ali_xy"],  key="Dloss_ali_xy",  outf=outf,niter=niter)
            plt.plot_loss_explicit(losses=self.losses["Gloss"],         key="Gloss",         outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_z"], key="Gloss_cycle_z", outf=outf,niter=niter)
            plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_xy"],key="Gloss_cycle_xy",outf=outf,niter=niter)
            plt.plot_loss_explicit(losses=self.losses["Gloss_ali_z"],   key="Gloss_ali_z",   outf=outf,niter=niter)
            plt.plot_loss_explicit(losses=self.losses["Gloss_ali_xy"],  key="Gloss_ali_xy",  outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["norm_grad"], key="norm_grad", outf=outf,niter=niter)
        # pdb.set_trace()
        # plt.plot_error(error,outf=outf)
        # del self.losses

        # tsave({'epoch':niter,'model_state_dict':self.F_.state_dict(),
        #     'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'./network/Fyx.pth')
        # tsave({'epoch':niter,'model_state_dict':self.Gy.state_dict(),
        #     'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'./network/Gy.pth')    
        # tsave({'model_state_dict':self.Dzd.state_dict()},'./network/Dzd_bb.pth')
        # tsave({'model_state_dict':self.Dy.state_dict()},'./network/Dy_bb.pth')    
        # tsave({'model_state_dict':self.Ddxz.state_dict()},'./network/Ddxz_bb.pth')

    # @profile
    def train(self):
        for t,a in self.strategy['tract'].items():
            if 'unique' in t.lower() and a:
                self.train_unique()

    # @profile            
    def generate(self):
        globals().update(self.cv)
        globals().update(opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'unique' in t and self.strategy['trplt']['unique']:
            # n = self.strategy['broadband']
            #plt.plot_generate_classic('broadband',Fyx,Gy,device,vtm,\
            #                          trn_loader,pfx="trn_set_bb",outf=outf)
            #plt.plot_generate_classic('broadband',Fyx,Gy,device,vtm,\
            #                          tst_loader,pfx="tst_set_bb",outf=outf)
            plt.plot_generate_classic('broadband',self.F_,self.Gy,device,vtm,\
                                      vld_loader,pfx="vld_set_bb_unique",opt=opt,outf=outf)
            #plt.plot_gofs(tag=['broadband'],Fef=self.Fef,Gx=self.Gx,Fyx=self.F_,\
            #             Gy=self.Gy,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
            #              pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb'},\
            #             outf=outf)
            # plt.plot_features('broadband',self.F_,self.Gy,nzd,device,vtm,vld_loader,pfx='set_bb',outf=outf)

        # if 'filtered' in t and self.strategy['trplt']['filtered']:
            # n = self.strategy['filtered']
            # Fef = deepcopy(self.Fef)
            # Gx = deepcopy(self.Gx)
            # if None not in n:
            #     print("Loading models {} {}".format(n[0],n[1]))
            #     Fef.load_state_dict(tload(n[0])['model_state_dict'])
            #     Gx.load_state_dict(tload(n[1])['model_state_dict'])
            #plt.plot_generate_classic('filtered',Fef,Gx,device,vtm,\
            #                          trn_loader,pfx="trn_set_fl",outf=outf)
            #plt.plot_generate_classic('filtered',Fef,Gx,device,vtm,\
            #                          tst_loader,pfx="tst_set_fl",outf=outf)
            plt.plot_generate_classic('filtered',self.F_,self.Gx,device,vtm,\
                                      vld_loader,pfx="vld_set_fl_unique",opt=opt, outf=outf)
            #plt.plot_gofs(tag=['filtered'],Fef=self.Fef,Gx=self.Gx,Fyx=self.F_,\
            #              Gy=self.Gy,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
            #              pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb'},\
            #              outf=outf)
            # plt.plot_features('filtered',self.Fef,self.Gx,nzf,device,vtm,vld_loader,pfx='set_fl',outf=outf)

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
            plt.plot_generate_hybrid(Fef,Gy,Ghz,device,vtm,\
                                      trn_loader,pfx="trn_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gy,Ghz,device,vtm,\
                                      tst_loader,pfx="tst_set_hb",outf=outf)
            plt.plot_generate_hybrid(Fef,Gy,Ghz,device,vtm,\
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
            plt.plot_compare_ann2bb(self.Fef,self.Gy,self.Ghz,device,vtm,\
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
                    Xd = Variable(xd_data).to(device) # BB-signal
                    Xf = Variable(xf_data).to(device) # LF-signal
                    zd = Variable(zd_data).to(device)
                    zf = Variable(zf_data).to(device)

    # @profile
    def _error(self,encoder, decoder, Xt, zt, dev):
        wnx,wnz,wn1 = noise_generator( Xt.shape,zt.shape,dev,rndm_args)
        X_inp = zcat(Xt,wnx.to(dev))
        ztr = encoder(X_inp).to(dev)
        # ztr = latent_resampling(Qec(X_inp),zt.shape[1],wn1)
        z_inp = zcat(ztr,wnz.to(dev))
        z_pre = zcat(zt,wnz.to(dev))
        Xr = decoder(z_inp)

        loss = torch.nn.MSELoss(reduction="mean")

        return loss(Xt,Xr)
