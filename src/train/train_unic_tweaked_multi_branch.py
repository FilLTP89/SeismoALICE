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
# from tools.generate_noise import lowpass_biquad
from database.database_sae import random_split 
# from tools.leave_p_out import k_folds
# from common.ex_common_setup import dataset2loader
from database.database_sae import thsTensorData
import torch.nn as nn 
import json
# import pprint as pp
import pdb
# from pytorch_summary import summary
# from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
# import GPUtil
# from torch.nn.parallel import DistributedDataParallel as Dnn.DataParallel
import time
import GPUtil
from database.toyset import Toyset, get_dataset
# app.RNDM_ARGS = {'mean': 0, 'std': 1.0}
from configuration import app

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
        b1 = 0.5
        b2 = 0.9999
        self.strategy = strategy
        #[TO DO ] si 
        self.ngpu     = min(torch.cuda.device_count()-1, opt.ngpu)

        nzd = opt.nzd
        ndf = opt.ndf

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # the follwings variable are the instance for the object Module from 
        # the package pytorch: torch.nn.modulese. /.
        # the names of variable are maped to the description aforementioned
        self.F_   = Module() # Single Encoder
        self.Gy   = Module() # Decoder for recorded signals
        self.Gx   = Module() # Decoder for synthetic signals
        self.Dy   = Module() # Discriminator for recorded signals
        self.Dx   = Module() # Discriminator for synthetic signals
        self.Dz   = Module() # Discriminator for latent space
        self.Dyz  = Module() # Discriminator for (y,zd)
        self.Dxz  = Module() # Discriminator for (x,zf)
        # Dyz(Dy(y),Dz(z))  /  Dxz(Dx(x),Dz(z))
        self.Dzz  = Module() # Discriminator for latent space reconstrution
        self.Dxx  = Module() # Discriminator for synthetic signal reconstrution
        self.Dyy  = Module() # Discriminator for recorded signal reconstruction
        self.Dzyx = Module() # Discriminator for recorded signal reconstruction

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
            factory = DataParalleleFactory()
        elif(cpus >1 and opt.ngpu >=1):
            print('DataParallele to be builded ...')
            factory = DataParalleleFactory()
            self.dp_mode = True
        else:
            print('environ not found')
        net = Network(factory)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'unique' in t:
            self.style='ALICE'
            # act = acts[self.style]
            n = self.strategy['unique']
            pdb.set_trace()

            self.F_  = net.Encoder(opt.config['F'],  opt).cuda()
            self.Gy  = net.Decoder(opt.config['Gy'], opt).cuda()
            self.Gx  = net.Decoder(opt.config['Gx'], opt).cuda()

            
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

                    # self.oGyx = RMSProp(ittc(self.F_.parameters(),
                    #     self.Gy.parameters(),
                    #     self.Gx.parameters()),
                    #     lr=glr,alpha=b2,
                    #     weight_decay=0.00001)
                self.optz.append(self.oGyx)

                ## Discriminators
                # pdb.!()
                # pdb.set_trace()
                self.Dy   = net.DCGAN_Dx(opt.config['Dy'],  opt).cuda()
                self.Dx   = net.DCGAN_Dx(opt.config['Dx'],  opt).cuda()
                self.Dzb  = net.DCGAN_Dz(opt.config['Dzb'], opt).cuda()
                self.Dzf  = net.DCGAN_Dz(opt.config['Dzf'], opt).cuda()
                self.Dyz  = net.DCGAN_DXZ(opt.config['Dyz'],opt).cuda()
                self.Dxz  = net.DCGAN_DXZ(opt.config['Dxz'],opt).cuda()
                self.Dzzb = net.DCGAN_Dz(opt.config['Dzzb'],opt).cuda()
                self.Dzzf = net.DCGAN_Dz(opt.config['Dzzf'],opt).cuda()
                self.Dxx  = net.DCGAN_Dx(opt.config['Dxx'], opt).cuda()
                self.Dyy  = net.DCGAN_Dx(opt.config['Dyy'], opt).cuda()
                self.Dzyx = net.DCGAN_Dz(opt.config['Dzyx'],opt).cuda()
                
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

                self.model =  nn.Sequential(*self.FGf,*self.Dnets)
                self.model =  nn.DataParallel(self.model).cuda()
                # print(self.model.F_)

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
        pdb.set_trace()
        print("Parameters of  Decoders/Decoders ")
        count_parameters(self.FGf)
        print("Parameters of Discriminators ")
        count_parameters(self.Dnets)
        self.bce_loss = BCE(reduction='mean')
        self.losses = {
            'Gloss':[0],'Dloss':[0],
                       'Gloss_ali_z':[0],
                       'Gloss_cycle_xy':[0],
                       'Gloss_ali_xy':[0],
                       'Gloss_ind':[0],
                       'Gloss_ftm':[0],
                       # 'Gloss_cycle_z':[0],
                       'Dloss_ali':[0],
                       'Dloss_ali_x':[0],
                       'Dloss_ali_y':[0],
                       'Dloss_ind':[0]
                       # 'Dloss_ali_xy':[0],
                       # 'Dloss_ali_z':[0]
                       }
        # pdb.set_trace()
    # def save_gradient_g_tb(self, net,tag,epoch): 
    #     for name, weight in net.named_parameters():
    #         self.writer.add_histogram('PARAM/{0}/{1}'.format(tag,name), weight, global_step=epoch)
    #         self.writer.add_histogram('PARAM/{0}/{1}.grad'.format(tag,name), weight.grad, global_step=epoch)
       
    # @profile
    def discriminate_xz(self,x,xr,z,zr):
        # Discriminate real
        pdb.set_trace()
        # torch.onnx.export(self.Dzf,zr,outf+"/Dzf.onnx")
        # torch.onnx.export(self.Dx,x,outf+"/Dx.onnx")
        ftz = self.Dzf(zr) #OK: no batchNorm
        ftx = self.Dx(x) #with batchNorm
        zrc = zcat(ftx,ftz)

        # zrc = zcat(ftx[0],ftz[0])
        # ftr = ftz[1]+ftx[1]

        # torch.save(self.Dxz,outf+"/Dxz.pth")exi
        ftxz = self.Dxz(zrc) #no batchNorm
        Dxz = ftxz
        # Dxz  = ftxz[0]
        # ftr += ftxz[1]

        # Discriminate fake
        ftz = self.Dzf(z)
        ftx = self.Dx(xr)
        zrc = zcat(ftx,ftz)

        # zrc = zcat(ftx[0],ftz[0])
        # ftf = ftz[1]+ftx[1]

        ftzx = self.Dxz(zrc)
        Dzx  = ftzx
        # Dzx  = ftzx[0]
        # ftf += ftzx[1]

        return Dxz,Dzx #,ftr,ftf

    def discriminate_yz(self,x,xr,z,zr):
        # Discriminate real
        pdb.set_trace()
        # torch.onnx.export(self.Dzb,zr,outf+"/Dzb.onnx")
        # torch.onnx.export(self.Dy,x,outf+"/Dy.onnx")
        ftz = self.Dzb(zr) #OK : no batchNorm
        ftx = self.Dy(x) #OK : with batchNorm

        # zrc = zcat(ftx[0],ftz[0]).reshape(-1,1)
        zrc = zcat(ftx,ftz)
        # zrc = zcat(ftx[0],ftz[0])
        # ftr = ftz[1]+ftx[1]
        # torch.onnx.export(self.Dyz,zrc,outf+"/Dyz.onnx")
        ftxz = self.Dyz(zrc) #OK : no batchNorm, don't forget the bias also
        Dxz  = ftxz
        # Dxz  = ftxz[0]
        # ftr += ftxz[1]
        
        # Discriminate fake
        ftz = self.Dzb(z)
        ftx = self.Dy(xr)
        
        # zrc = zcat(ftx[0],ftz[0]).reshape(-1,1)
        zrc = zcat(ftx,ftz)
        # zrc = zcat(ftx[0],ftz[0])
        # ftf = ftz[1]+ftx[1]

        ftzx = self.Dyz(zrc)
        Dzx  = ftzx
        # Dzx  = ftzx[0]
        # ftf += ftzx[1]

        return Dxz,Dzx 
    
    ''' Methode that discriminate real and fake hybrid signal type'''
    # def discriminate_hybrid_xz(self,Xd,Xdr,zd,zdr):
    #     # pdb.set_trace()
    #     # Discriminate real

    #     ftz = self.Dzf(zdr)
    #     ftX = self.Dy(Xd)
    #     zrc = zcat(ftX[0],ftz[0])
    #     ftr = ftz[1]+ftX[1]
    #     ftXz = self.Dhzdzf(zrc)
    #     DXz  = ftXz[0]
    #     ftr += ftXz[1]
        
    #     # Discriminate fake
    #     ftz = self.Dzf(zd)
    #     ftX = self.Dy(Xdr)
    #     zrc = zcat(ftX[0],ftz[0])
    #     ftf = ftz[1]+ftX[1]
    #     ftzX = self.Dhzdzf(zrc)
    #     DzX  = ftzX[0]
    #     ftf += ftzX[1]
        
    #     return DXz,DzX,ftr,ftf

    # @profile
    def discriminate_xx(self,x,xr):
        # pdb.set_trace()
        # torch.onnx.export(self.Dxx,zcat(x,x),outf+"/Dxx.onnx")
        # x et xr doivent avoir la même distribution !
        Dreal = self.Dxx(zcat(x,x))#with batchNorm
        Dfake = self.Dxx(zcat(x,xr))
        return Dreal,Dfake

    def discriminate_yy(self,x,xr):
        # pdb.set_trace()
        # torch.onnx.export(self.Dyy,zcat(x,x),outf+"/Dyy.onnx")
        Dreal = self.Dyy(zcat(x,x )) #with batchNorm
        Dfake = self.Dyy(zcat(x,xr))
        return Dreal,Dfake

    @profile
    def discriminate_zzb(self,z,zr):
        # pdb.set_trace()
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
        # self.F_.eval()  , self.Gx.eval()  , self.Gy.eval()
        # self.Dy.train() , self.Dx.train() , self.Dz.train()
        # self.Dyz.train(), self.Dxz.train(), self.Dzzb.train()
        # self.Dyy.train(), self.Dxx.train(), self.Dzzf.train()
        # 0. Generate noise
        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        wny,wnzd,_ = noise_generator(y.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)

        # wny = wny.to(y.device)
        # wnx = wnx.to(x.device)
        # wnzd = wnzd.to(zd.device)
        # wnzf = wnzf.to(zf.device)

        # pdb.set_trace()

        # 1. Concatenate inputs
        # y_inp  = zcat(y,wny)
        # x_inp  = zcat(x,wnx)
        # zd_inp = zcat(zd,wnzd)
        # zf_inp = zcat(zf,wnzf)
        
        # 2. Generate conditional samples
        y_gen = self.Gy(zcat(zd,wnzd))
        x_gen = self.Gx(zcat(zf,wnzf))

        # zd_gen,_  = self.F_(y_inp)
        # _, zf_gen = self.F_(x_inp)

        # F(y)|y,F(y)|yx,F(y)|x
        zdd_gen,zdf_gen,*other = self.F_(zcat(y,wny)) # zdd_gen = zy, zdf_gen = zyx
        # F(x)|y,F(x)|yx,F(x)|x
        _,zfd_gen, *other = self.F_(zcat(x,wnx)) # zff_gen = zx, zfd_gen = zxy
 
        # [F(y)|yx,F(y)|y]
        zd_gen = zcat(zdf_gen,zdd_gen) # zy generated by y
        # [F(x)|yx,F(x)|x]
        zf_gen = zcat(zfd_gen) # zx generated by x
        # zf_gen = zfd_gen # zx generated by x

        # self.writer.add_histogram('Latent/broadband',zd_gen)
        # self.writer.add_histogram('Latent/filtered',zf_gen)
        
        D_zyx,D_zxy = self.discriminate_zxy(zdf_gen,zfd_gen)
        #zd_ind = zcat(zfd_gen,zfd_ind) # zy generated by x
        #zf_ind = zcat(zdf_gen,zdf_ind) # zx generated by y
        ## concatenate : first common part zxy and then zy or zx
        ## ici il faut les discriminateurs z
        #Dreal_zzy,Dfake_zzy = self.discriminate_zzb(zd,zd_ind)
        #Dreal_zzx,Dfake_zzx = self.discriminate_zzf(zf,zf_ind)

        # Dloss_ind_y = self.bce_loss(Dreal_zzy,o1l(Dreal_zzy))+\
        #     self.bce_loss(Dfake_zzy,o0l(Dfake_zzy))+\
        # Dloss_ind_x = self.bce_loss(Dreal_zzx,o1l(Dreal_zzx))+\
        #     self.bce_loss(Dfake_zzx,o0l(Dfake_zzx))
        Dloss_ind = -torch.mean(ln0c(D_zyx)+ln0c(1.0-D_zxy))
        #Dloss_ind_y = -torch.mean(ln0c(Dreal_zzy)+ln0c(1.0-Dfake_zzy))
        #Dloss_ind_x = -torch.mean(ln0c(Dreal_zzx)+ln0c(1.0-Dfake_zzx))
        # Dloss_ind_y = -(torch.mean(Dreal_zzy)-torch.mean(Dfake_zzy))
        # Dloss_ind_x = -(torch.mean(Dreal_zzx)-torch.mean(Dfake_zzx))
        #Dloss_ind =  Dloss_ind_x + Dloss_ind_y
        # 3. Cross-Discriminate XZ
        Dyz,Dzy = self.discriminate_yz(y,y_gen,zd,zd_gen)
        Dxz,Dzx = self.discriminate_xz(x,x_gen,zf,zf_gen)
        # pdb.set_trace()
        # 4. Compute ALI discriminator loss
        # Dloss_ali_y = self.bce_loss(Dzy,o1l(Dzy)) + self.bce_loss(Dyz,o0l(Dyz))
        # Dloss_ali_x = self.bce_loss(Dzx,o1l(Dzx)) + self.bce_loss(Dxz,o0l(Dxz))
        Dloss_ali_y = -torch.mean(ln0c(Dzy)+ln0c(1.0-Dyz))
        Dloss_ali_x = -torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        Dloss_ali = Dloss_ali_x + Dloss_ali_y

        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        wny,wnzd,_ = noise_generator(y.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)

        # wny  = wny.to(y.device)
        # wnx  = wnx.to(x.device)
        # wnzd = wnzd.to(zd.device)
        # wnzf = wnzf.to(zf.device)

        # 1. Concatenate inputs
        # y_gen  = zcat(y_gen,wny)
        # x_gen  = zcat(x_gen,wnx)
        # zd_gen = zcat(zd_gen,wnzd)
        # zf_gen = zcat(zf_gen,wnzf)
        
        # pdb.set_trace()
        # 2. Generate reconstructions
        y_rec  = self.Gy(zcat(zd_gen,wnzd))
        x_rec  = self.Gx(zcat(zf_gen,wnzf))

        zdd_gen,zdf_gen,*other = self.F_(zcat(y_gen,wny))
        _,zfd_gen,*other = self.F_(zcat(x_gen,wnx))

        zd_rec = zcat(zdf_gen,zdd_gen) # zy generated by y
        zf_rec = zcat(zfd_gen) # zx generated by x
        # zf_rec = zfd_gen # zx generated by x
 
        #zd_ind = zcat(zfd_gen,zfd_ind) # zy generated by x
        #zf_ind = zcat(zdf_gen,zdf_ind) # zx generated by y

        # zd_rec = self.F_(y_gen)
        # zf_rec = self.F_(x_gen)
        # z_rec = latent_resampling(self.Fef(X_gen),nzf,wn1)

        # 3. Cross-Discriminate XX
        Dreal_y,Dfake_y = self.discriminate_yy(y,y_rec)
        Dreal_x,Dfake_x = self.discriminate_xx(x,x_rec)

        # Dloss_rec_y = self.bce_loss(Dreal_y,o1l(Dreal_y))+self.bce_loss(Dfake_y,o0l(Dfake_y))
        # Dloss_rec_x = self.bce_loss(Dreal_x,o1l(Dreal_x))+self.bce_loss(Dfake_x,o0l(Dfake_x))
        Dloss_rec_y = -torch.mean(ln0c(Dreal_y)+ln0c(1.0-Dfake_y))
        Dloss_rec_x = -torch.mean(ln0c(Dreal_x)+ln0c(1.0-Dfake_x))
        # Dloss_rec_y = -(torch.mean(Dreal_y)-torch.mean(Dfake_y))
        # Dloss_rec_x = -(torch.mean(Dreal_x)-torch.mean(Dfake_x))

        # warning in the perpose of us ing the BCEloss the value should be greater than zero, 
        # so we apply a boolean indexing. BCEloss use log for calculation so the negative value
        # will lead to isses. Why do we have negative value? the LeakyReLU is not tranform negative value to zero
        # I recommanded using activation's function that values between 0 and 1, like sigmoid
        # pdb.set_trace()
        # Dloss_ali_xy = self.bce_loss(Dreal_y,o1l(Dreal_y))+self.bce_loss(Dfake_y,o1l(Dfake_y))+\
        #     self.bce_loss(Dreal_x,o1l(Dreal_x))+self.bce_loss(Dfake_x,o1l(Dfake_x))
        #4. Cross-Discriminate ZZ
        # pdb.set_trace()
        Dreal_zd,Dfake_zd = self.discriminate_zzb(zd,zd_rec)
        Dreal_zf,Dfake_zf = self.discriminate_zzf(zf,zf_rec)
        # Dreal_z, Dfake_z  = self.discriminate_zz(zf,zd)
        
        # Dloss_rec_z = self.bce_loss(Dreal_zd,o1l(Dreal_zd))+self.bce_loss(Dfake_zd,o0l(Dfake_zd))+\
        #     self.bce_loss(Dreal_zf,o1l(Dreal_zf))+self.bce_loss(Dfake_zf,o0l(Dfake_zf))
            # self.bce_loss(Dreal_z,o1l(Dreal_z))+self.bce_loss(Dfake_z,o0l(Dfake_z))
        Dloss_rec_zy = -torch.mean(ln0c(Dreal_zd)+ln0c(1.0-Dfake_zd))
        Dloss_rec_zx = -torch.mean(ln0c(Dreal_zf)+ln0c(1.0-Dfake_zf))

        # Total loss
        Dloss_rec = Dloss_rec_x+Dloss_rec_y+Dloss_rec_zx+Dloss_rec_zy

        Dloss = Dloss_ali + Dloss_ind + Dloss_rec

        Dloss.backward()
        self.oDyxz.step()#,clipweights(self.Dnets),
        zerograd(self.optz)
        self.losses['Dloss'].append(Dloss.tolist())  
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())  
        self.losses['Dloss_ali_x'].append(Dloss_ali_x.tolist())
        self.losses['Dloss_ali_y'].append(Dloss_ali_y.tolist())
        self.losses['Dloss_ind'].append(Dloss_ind.tolist())
        # self.losses['Dloss_ali_xy'].append(Dloss_ali_xy.tolist())  
        # self.losses['Dloss_ali_z'].append(Dloss_ali_z.tolist())

    # @profile
    def alice_train_generator_adv(self,y,zd,x,zf,zd_fix, zf_fix):
        # Set-up training
        zerograd(self.optz)
        modalite(self.FGf, mode = 'train')
        modalite(self.Dnets, mode ='train')
        # self.F_.train() , self.Gx.train() , self.Gy.train()
        # self.Dy.train() , self.Dx.train() , self.Dz.train()
        # self.Dyz.train(), self.Dxz.train() ,self.Dzzb.train()
        # self.Dyy.train(), self.Dxx.train(), self.Dzzf.train()
        # l1 = torch.nn.L1Loss()
        # 0. Generate noise
        wny,wnzd,_ = noise_generator(y.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
         
        # 1. Concatenate inputs
        # y_inp  = zcat(y,wny)
        # x_inp  = zcat(x,wnx)
        # zd_inp = zcat(zd,wnzd)
        # zf_inp = zcat(zf,wnzf)

        # pdb.set_trace()
         
        # 2. Generate conditional samples
        y_gen = self.Gy(zcat(zd,wnzd)) #(100,64,64)->(100,3,4096)
        x_gen = self.Gx(zcat(zf,wnzf))
        # zd_gen = self.F_(y_inp)[:,:zd.shape[1]] #(100,64,64) -> (100,32,64)
       # pdb.set_trace()
        # zd_gen, _ = self.F_(y_inp)
        # _, zf_gen = self.F_(x_inp)
        zdd_gen,zdf_gen,*other = self.F_(zcat(y,wny)) # zdd_gen = zy, zdf_gen = zyx
        _,zfd_gen, *other = self.F_(zcat(x,wnx)) # zff_gen = zx, zfd_gen = zxy
 
        zd_gen = zcat(zdf_gen,zdd_gen) # zy generated by y
        zf_gen = zcat(zfd_gen) # zx generated by x
        # zf_gen = zfd_gen # zx generated by x
 
        #zd_ind = zcat(zfd_gen,zfd_ind) # zy generated by x
        #zf_ind = zcat(zdf_gen,zdf_ind) # zx generated by y

        # ici il faut les discriminateurs z
        D_zyx,D_zxy = self.discriminate_zxy(zdf_gen,zfd_gen)
        #Dreal_zzy,Dfake_zzy = self.discriminate_zzb(zd,zd_ind)
        #Dreal_zzx,Dfake_zzx = self.discriminate_zzf(zf,zf_ind)
        Gloss_ind = torch.mean(-D_zxy+D_zyx)
        #Gloss_ind_y = self.bce_loss(Dfake_zzy,o1l(Dfake_zzy))
        #Gloss_ind_x = self.bce_loss(Dfake_zzx,o1l(Dfake_zzx))

        # Ici on force le F a générer des zxy égales pour F_(x) et F_(y)
        #Gloss_ind_cycle = torch.mean(torch.abs(zdf_gen-zfd_gen))
        
        #Gloss_ind = Gloss_ind_x+Gloss_ind_y+Gloss_ind_cycle
        # zf_gen = zcat(zf_gen,wnzf) # = dim zd_gen
        # zd_gen = self.F_(y_inp)
        # zf_gen = self.F_(x_inp)
        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
         
        # 3. Cross-Discriminate XZ
        # pdb.set_trace()
        Dyz,Dzy = self.discriminate_yz(y,y_gen,zd,zd_gen)
        Dxz,Dzx = self.discriminate_xz(x,x_gen,zf,zf_gen)

        # 4. Compute ALI Generator loss 
        Gloss_ali = torch.mean(-Dyz+Dzy)+torch.mean(-Dxz+Dzx)
        Gloss_ftm = 0.0
        # for rf,ff in zip(ftxz,ftzx):
        #     Gloss_ftm += torch.mean((rf-ff)**2)
        # for rf,ff in zip(ftyz,ftzy):
        #     Gloss_ftm += torch.mean((rf-ff)**2)

        # 0. Generate noise
        wny,wnzd,_ = noise_generator(y.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
        wnx,wnzf,_ = noise_generator(x.shape,zf.shape,app.DEVICE,app.RNDM_ARGS)
        
        # 1. Concatenate inputs
        y_gen  = zcat(y_gen,wny)
        x_gen  = zcat(x_gen,wnx) 
        zd_gen = zcat(zd_gen,wnzd) #(100,64,64)
        zf_gen = zcat(zf_gen,wnzf) #(100,32,64)

        # 2. Generate reconstructions
        y_rec = self.Gy(zd_gen)
        x_rec = self.Gx(zf_gen)
        # zd_rec = self.F_(y_gen)[:,:zd.shape[1]]

        zdd_gen,zdf_gen,*other = self.F_(y_gen)
        _,zfd_gen,*other = self.F_(x_gen)

        zd_rec = zcat(zdf_gen,zdd_gen) # zy generated by y
        zf_rec = zcat(zfd_gen) # zx generated by x
 
        # zd_ind = zcat(zfd_gen,zfd_ind) # zy generated by x
        # zf_ind = zcat(zdf_gen,zdf_ind) # zx generated by y
    
        # pdb.set_trace()
        # 3. Cross-Discriminate XX
        _,Dfake_y = self.discriminate_yy(y,y_rec)
        _,Dfake_x = self.discriminate_xx(x,x_rec)
        # penality 
        penalty = 1.0
        # Gloss_ali_xy = self.bce_loss(Dfake_y,o1l(Dfake_y))+penalty*self.bce_loss(Dfake_x,o1l(Dfake_x))
        Gloss_ali_xy = torch.mean(Dfake_y + Dfake_x)
        Gloss_cycle_xy = torch.mean(torch.abs(y-y_rec))+torch.mean(torch.abs(x-x_rec))
        # Gloss_cycle_xy = torch.mean((y-y_rec)**2)+torch.mean((x-x_rec)**2)
#         for rf,ff in zip(ftX_real,ftX_fake):
#             Gloss_ftmX += torch.mean((rf-ff)**2)
        
        # 4. Cross-Discriminate ZZ
        # pdb.set_trace()
        _,Dfake_zd = self.discriminate_zzb(zd,zd_rec)
        # pdb.set_trace()
        _,Dfake_zf = self.discriminate_zzf(zf,zf_rec)
        # Gloss_ali_z = self.bce_loss(Dfake_zd,o1l(Dfake_zd))+self.bce_loss(Dfake_zf,o1l(Dfake_zf))
        Gloss_ali_z   = torch.mean(Dfake_zd + Dfake_zf)
        Gloss_cycle_z = torch.mean(torch.abs(zd_fix-zd_rec))+torch.mean(torch.abs(zf_fix-zf_rec))
#         Gloss_ftmz = 0.
#         for rf,ff in zip(ftz_real,ftz_fake):
#             Gloss_ftmz += torch.mean((rf-ff)**2)    
        # Total Loss
        # pdb.set_trace()
        Gloss = Gloss_ali_xy+Gloss_cycle_xy+Gloss_ali_z+Gloss_ind + Gloss_cycle_z #+Gloss_ftm
        # Gloss = (Gloss_ftm*0.7)*(1.-0.7)/2.0 + \
        #     (Gloss_cycle_xy*0.9+Gloss_ali_xy*0.1)*(1.-0.1)/1.0 +\
        #     (Gloss_ali_z*0.7)*(1.-0.7)/2.0
        Gloss.backward()
        self.oGyx.step()
        zerograd(self.optz)
         
        self.losses['Gloss'].append(Gloss.tolist()) 
        # self.losses['Gloss_ftm'].append(Gloss_ftm.tolist())
        self.losses['Gloss_cycle_xy'].append(Gloss_cycle_xy.tolist())
        self.losses['Gloss_ali_z'].append(Gloss_ali_z.tolist())
        self.losses['Gloss_ali_xy'].append(Gloss_ali_xy.tolist())
        self.losses['Gloss_ind'].append(Gloss_ind.tolist())
        
    # @profile
    def train_unique(self):
        print('Training on both recorded and synthetic signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        error = {}

        start_time = time.time()
        verbose = False

        # dataset = Toyset(nsy = 1280)
        # trn_loader, vld_loader =  get_dataset(dataset, rank = 0, 
        #     batch_size = 256, nsy = 1280, world_size = 1)

        # total_step = len(trn_loader)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
            # for b,batch in enumerate(self.trn_loader):
                # pdb.set_trace()
                # y,x,_ = batch
                y, x, zd, zf, *other = batch
                y   = y.to(app.DEVICE) # recorded signal
                x   = x.to(app.DEVICE) # synthetic signal
                zd  = zd.to(app.DEVICE)
                zf  = zf.to(app.DEVICE)
                # Train G/D
                wnzd = torch.randn(*zd.shape).to(app.DEVICE)
                wnzf = torch.randn(*zf.shape).to(app.DEVICE)
                
                for _ in range(5):
                    self.alice_train_discriminator_adv(y,wnzd,x,wnzf)                 
                for _ in range(1):
                    self.alice_train_generator_adv(y,wnzd,x,wnzf,zd,zf)

                if verbose:
                    t = (time.time() - start_time)/60
                    print(f'Epoch [{epoch}/{opt.niter}]\tStep [{b}/{total_step-1}] \ttime {t}')

                #     print("Outside: input size", y.size())
                #     wny,wnz,_ = noise_generator(y.shape,zd.shape,app.DEVICE,app.RNDM_ARGS)
                #     print("Outside: output_size", self.F_(zcat(y,wny))[0].size())


            if epoch%10== 0:
                print("--- {} minutes ---".format((time.time() - start_time)/60))

            
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)

            if verbose:
                GPUtil.showUtilization(all=True)

            
            if save_checkpoint and verbose:
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
                   
        
        # plt.plot_loss_explicit(losses=self.losses["Dloss_ali"],     key="Dloss_ali",     outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["Dloss_ali_x"],   key="Dloss_ali_x",   outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["Dloss_ali_y"],   key="Dloss_ali_y",   outf=outf,niter=niter)
        # # plt.plot_loss_explicit(losses=self.losses["Dloss_ali_xy"],  key="Dloss_ali_xy",  outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["Gloss"],         key="Gloss",         outf=outf,niter=niter)
        # # plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_z"], key="Gloss_cycle_z", outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_xy"],key="Gloss_cycle_xy",outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["Gloss_ali_z"],   key="Gloss_ali_z",   outf=outf,niter=niter)
        # plt.plot_loss_explicit(losses=self.losses["Gloss_ali_xy"],  key="Gloss_ali_xy",  outf=outf,niter=niter)
    

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
            plt.plot_generate_classic('broadband',self.F_,self.Gy,app.DEVICE,vtm,\
                                      vld_loader,pfx="vld_set_bb_unique",opt=opt,outf=outf)
            
            plt.plot_generate_classic('filtered',self.F_,self.Gx,app.DEVICE,vtm,\
                                      vld_loader,pfx="vld_set_fl_unique",opt=opt, outf=outf)

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


