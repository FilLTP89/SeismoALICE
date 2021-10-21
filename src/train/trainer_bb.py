# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''
# import warnings
# warnings.filterwarnings("ignore")
from copy import deepcopy
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
import torch.distributed as dist
import numpy as np
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
    def __init__(self, cv):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        # And therefore this latter is become accessible to the methods in this class
        # globals().update(cv)
        # define as global opt and passing it as a dictonnary here
        globals().update(cv)
        globals().update(opt.__dict__)
        self.opt = opt

        super(trainer, self).__init__()
        self.gpu = ngpu - 1
        # pdb.set_trace()
        print(f'prepare to train on {self.gpu}')
        print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')
        # rank = self.opt.nr * self.opt.ngpu + self.gpu
        # rank = args.nr * args.gpus + gpu

        # dist.init_process_group(
        #     backend='nccl',
        #     init_method='env://',
        #     world_size=opt.world_size,
        #     rank=rank)
        # process_group = torch.distributed.new_group()
        # seed = 0
        # torch.cuda.set_device(self.gpu)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # np.random.seed(0)

        # if rank == 0:
            #toy test :
        dataset = Toyset(nsy = 1280)
        self.batch_size = opt.batchSize
        # if rank == 0:
        #     get_dataset(dataset, rank = 0, batch_size = self.batch_size, 
        #         nsy = 1280, world_size = 1)

        self.trn_loader, self.vld_loader = get_dataset(dataset, rank = 0, batch_size = self.batch_size, 
                nsy = 1280, world_size = 1)
        # train_set, vld_set = torch.utils.data.random_split(dataset, [80,20])

        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         train_set,
        #         num_replicas=self.opt.world_size,
        #         rank=rank)

        # vld_sampler = torch.utils.data.distributed.DistributedSampler(
        #         vld_set,
        #         num_replicas=self.opt.world_size,
        #         rank=rank)

        # self.trn_loader = torch.utils.data.DataLoader(dataset=train_set, 
        #             batch_size =self.batch_size, 
        #             shuffle    =False,
        #             num_workers=0,
        #             pin_memory =True,
        #             sampler    = train_sampler)

        # self.vld_loader = torch.utils.data.DataLoader(dataset=vld_set, 
        #             batch_size =self.batch_size, 
        #             shuffle    =False,
        #             num_workers=0,
        #             pin_memory =True,
        #             sampler    = vld_sampler)
        # dist.barrier()
        # self.cv = cv
        self.gr_norm = []
        # define as global variable the cv object. 
        

        # passing the content of file ./strategy_bb_*.txt
        self.strategy=strategy
        self.ngpu = ngpu

        nzd = self.opt.nzd
        ndf = self.opt.ndf
        
        # the follwings variable are the instance for the object Module from 
        # the package pytorch: torch.nn.modulese. 
        # the names of variable are maped to the description aforementioned
        self.Fed  = Module()
        self.Gdd  = Module()
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
        self.oGdxz  = None
        self.oGfxz  = None
        self.oGhxz  = None
        self.dp_mode= True
        
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        """
            This part is for training with the broadband signal
        """
        #we determine in which kind of environnement we are 
        # pdb.set_trace()
        #we are getting number of cpus 
        # breakpoint()
        cpus  =  int(os.environ.get('SLURM_NPROCS'))
        #we determine in which kind of environnement we are 
        if(cpus ==1 and self.opt.ngpu >=1):
            print('ModelParallele to be builded ...')
            self.dp_mode = False
            factory = DataParalleleFactory()
        elif(cpus >1 and self.opt.ngpu >=1):
            print('DataParallele to be builded ...')
            factory = DataParalleleFactory()
            self.dp_mode = True
        else:
            print('environ not found')
        net = Network(factory)

        if 'broadband' in t:
            self.style='ALICE'
            # act = acts[self.style]

            flagF = True
            n = self.strategy['broadband']
            print("Loading broadband generators")

            # if rank == 0:
            self.Fef = net.Encoder(self.opt.config['encoder'],self.opt).cuda()
            self.Gdf = net.Decoder(self.opt.config['decoder'], self.opt).cuda()
            # dist.barrier(), find_unused_parameters=True

            # self.Fef = DDP(self.Fef, device_ids=[self.gpu], find_unused_parameters=True)

            # self.Gdf = DDP(self.Gdf, device_ids=[self.gpu], find_unused_parameters=True)

            # if rank == 0:
            #     print(self.Fef)
            # t = torch.randn(10, 6, 4096).to(self.gpu)
            # print("shape of t", t.shape)
            # print("shape of F(t)",self.Fef(t).shape)
            if self.strategy['tract']['broadband']:
                if None in n:        
                    self.FGf = [self.Fef,self.Gdf]
                    self.oGfxz = reset_net(self.FGf,func=set_weights,
                        lr=glr*self.opt.world_size,b1=b1,b2=b2,
                        weight_decay=0.00001)
                else:
                    print("broadband generators: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0])['model_state_dict']).cuda()
                    self.Gdf.load_state_dict(tload(n[1])['model_state_dict']).cuda() 
                    self.oGfxz = Adam(ittc(self.Fef.parameters(),self.Gdf.parameters()),
                                      lr=glr,betas=(b1,b2),weight_decay=0.00001).cuda()
                self.optzf.append(self.oGfxz)
                # if rank == 0:
                self.Dszf = net.DCGAN_Dz(self.opt.config['Dszf']  , self.opt).cuda()
                self.DsXf = net.DCGAN_Dx(self.opt.config['DsXf']  , self.opt).cuda()
                self.Dfxz = net.DCGAN_DXZ(self.opt.config['Dfxz'] , self.opt).cuda()
                # dist.barrier()

                # self.Dszf = DDP(self.Dszf, device_ids=[self.gpu],
                #     find_unused_parameters=True)

                # self.DsXf = DDP(self.DsXf,device_ids=[self.gpu],
                #     find_unused_parameters=True)

                # self.Dfxz = DDP(self.Dfxz, device_ids=[self.gpu],
                #     find_unused_parameters=True)
                self.Dfnets.append(self.DsXf)
                self.Dfnets.append(self.Dszf)
                self.Dfnets.append(self.Dfxz)


                # if rank ==0:
                self.Dsrzf = net.DCGAN_Dz(self.opt.config['Dsrzf'], self.opt).cuda()
                self.DsrXf = net.DCGAN_Dx(self.opt.config['DsrXf'], self.opt).cuda()
                # dist.barrier()

                # self.Dsrzf = DDP(self.Dsrzf, device_ids=[self.gpu],
                #     find_unused_parameters=True)
                # self.DsrXf = DDP(self.DsrXf, device_ids=[self.gpu],
                #     find_unused_parameters=True)
                
                self.Dfnets.append(self.DsrXf)
                self.Dfnets.append(self.Dsrzf)

                self.oDfxz = reset_net(self.Dfnets,func=set_weights,
                                lr=rlr * opt.world_size,optim='rmsprop')
                self.optzf.append(self.oDfxz)

                # pdb.set_trace()

            else:
                if None not in n:
                    print("broadband generators - no train: {0} - {1}".format(*n))
                    self.Fef.load_state_dict(tload(n[0])['model_state_dict']).cuda()
                    self.Gdf.load_state_dict(tload(n[1])['model_state_dict']).cuda()
                else:
                    flagF=False

        
        self.bce_loss = BCE(reduction='mean').to(ngpu-1)
        self.losses = {'Dloss_t':[0],'Dloss_f':[0],'Dloss_t':[0],
                       'Gloss_x':[0],'Gloss_z':[0],'Gloss_t':[0],
                       'Gloss_xf':[0],'Gloss_xt':[0],'Gloss_f':[0],
                       'Gloss':[0],'Dloss':[0],'Gloss_ftm':[0],'Gloss_ali_X':[0],
                       'Gloss_ali_z':[0],'Gloss_cycle_X':[0],
                       'Gloss_cycle_z':[0],'Dloss_ali':[0],
                       'Dloss_ali_X':[0],'Dloss_ali_z':[0], 'norm_grad':[0]}
    
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
        # Set-up training
        zerograd(self.optzf)
        
        self.Fef.eval(),self.Gdf.eval()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)
        #make sure that noises are at the same device as data
        # wnx = wnx.to(self.gpu)
        # wnz = wnz.to(self.gpu)

        # t = torch.randn(10, 6, 4096).to(self.gpu)
        # print("shape of t", t.shape)
        # print("shape of F(t) :",self.Fef)

        # 1. Concatenate inputs
        X_inp = zcat(Xf,wnx)
        z_inp = zcat(zf,wnz)
        
        
        # 2. Generate conditional samples
        # print(" device Fef : ", self.Fef)
        X_gen = self.Gdf(z_inp)
        z_gen = self.Fef(X_inp)        # print("X_gen : ",X_gen.shape)


        # print("X_inp : ",X_inp.shape)
        # print("z_gen : ",z_gen.shape)

        # 3. Cross-Discriminate XZ
        DXz,DzX = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)

        # Dloss_ftm = 0.
        # for rf,ff in zip(ftXz,ftzX):
        #     Dloss_ftm += torch.mean((rf-ff)**2)
         
        # 4. Compute ALI discriminator loss
        Dloss_ali = -torch.mean(ln0c(DzX)+ln0c(1.0-DXz))
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)
        
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx)
        z_gen = zcat(z_gen,wnz)
        
        # pdb.set_trace()
        # 2. Generate reconstructions
        X_rec = self.Gdf(z_gen)
        z_rec = self.Fef(X_gen) 
        # 3. Cross-Discriminate XX
        
        Dreal_X,Dfake_X = self.discriminate_filtered_xx(Xf,X_rec)

        # warning in the perpose of using the BCEloss the value should be greater than zero, 
        # so we apply a boolean indexing. BCEloss use log for calculation so the negative value
        # will lead to isses. Why do we have negative value? the LeakyReLU is not tranform 
        #negative value to zero
        # I recommanded using activation's function that values between 0 and 1, like sigmoid
        
        Dloss_ali_X = self.bce_loss(Dreal_X,o1l(Dreal_X))+\
                      self.bce_loss(Dfake_X,o1l(Dfake_X))
        Dloss_ali_X = Dloss_ali_X
            
        # 4. Cross-Discriminate ZZ
        # pdb.set_trace()
        Dreal_z,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)

        Dloss_ali_z = self.bce_loss(Dreal_z,o1l(Dreal_z))+\
                      self.bce_loss(Dfake_z,o0l(Dfake_z))

        # Total loss
        Dloss = Dloss_ali + 10.*Dloss_ali_X + 100.*Dloss_ali_z
        # Dloss = Dloss_ali + Dloss_ali_X + Dloss_ali_z
        Dloss.backward()
        self.oDfxz.step(),clipweights(self.Dfnets),zerograd(self.optzf)
        self.losses['Dloss'].append(Dloss.tolist())  
        self.losses['Dloss_ali'].append(Dloss_ali.tolist())  
        self.losses['Dloss_ali_X'].append(Dloss_ali_X.tolist())  
        self.losses['Dloss_ali_z'].append(Dloss_ali_z.tolist())
        
    def alice_train_broadband_generator_adv_xz(self,Xf,zf):
#OK
        # Set-up training
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

        

        # print(self.Fef)
        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
         
        # 3. Cross-Discriminate XZ
        DXz,DzX = self.discriminate_filtered_xz(Xf,X_gen,zf,z_gen)

        # 4. Compute ALI Generator loss
        Gloss_ali = torch.mean(-DXz+DzX)

        # for rf,ff in zip(ftXz,ftzX):
        #     Gloss_ftm = Gloss_ftm + torch.mean((rf-ff)**2)
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,self.gpu,rndm_args)
        
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
        
        # 4. Cross-Discriminate ZZ
        Dreal_z,Dfake_z = self.discriminate_filtered_zz(zf,z_rec)

        Gloss_ali_z = self.bce_loss(Dfake_z,o1l(Dfake_z)) +self.bce_loss(Dreal_z,o1l(Dreal_z))

        # Gloss_cycle_z = torch.mean(torch.abs(zf-z_rec)**2)
        Gloss_cycle_z = l2(zf,z_rec)

        # Total Loss
        # pdb.set_trace()

        Gloss = Gloss_ali +Gloss_cycle_X+Gloss_ali_X + Gloss_cycle_z+Gloss_ali_z
        Gloss.backward(),self.oGfxz.step(),zerograd(self.optzf)
         
        self.losses['Gloss'].append(Gloss.tolist()) 
        self.losses['Gloss_ftm'].append(Gloss_ali_X.tolist())
        self.losses['Gloss_ali_X'].append(Gloss_ali_X.tolist())
        self.losses['Gloss_ali_z'].append(Gloss_ali_z.tolist())
        self.losses['Gloss_cycle_X'].append(Gloss_cycle_X.tolist())
        self.losses['Gloss_cycle_z'].append(Gloss_cycle_z.tolist())

    def train_broadband(self):
#OK
        print("[!] In function train broadband ... ")
        # globals().update(self.cv)
        globals().update(self.opt.__dict__)
        error = {}
        start_time = time.time()
        # breakpoint()
        total_step = len(self.trn_loader)
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(niter):
                for b,batch in enumerate(self.trn_loader):
                    # pdb.set_trace()
                    # Load batch
                    # place = self.opt.dev if self.dp_mode else ngpu-1
                    xd_data,_,zd_data,*other = batch
                    # print(zd_data.shape)
                    # xd_data,_,zd_data,_,_,_,_ = batch
                    Xf = Variable(xd_data).to(self.gpu, non_blocking = True) # LF-signal
                    zf = Variable(zd_data).to(self.gpu, non_blocking = True)
                    # t = torch.randn(10,6,4096).to(self.gpu)
                    # print("F(x) ",  self.Fef(t).shape)
                    # print("Xf and zf", Xf.shape, zf.shape)
    #               # Train G/D
                    for _ in range(5):
                        self.alice_train_broadband_discriminator_adv_xz(Xf,zf)
                        # torch.cuda.empty_cache()
                    for _ in range(1):
                        self.alice_train_broadband_generator_adv_xz(Xf,zf)
                        # torch.cuda.empty_cache()

                    # err = self._error(self.Fef, self.Gdf, Xf,zf,self.gpu)
                    # a =  err.cpu().data.numpy().tolist()
                    # #error in percentage (%)
                    # if b in error:
                    #     error[b] = np.append(error[b], a)
                    # else:
                    #     error[b] = a
                    # if self.gpu == 0:
                    str0 = 'Epoch [{}/{}]\tStep [{}/{}]'.format(epoch,self.opt.niter,b,total_step)
                    print(str0)

                print("--- {} minutes ---".format((time.time() - start_time)/60))
                # GPUtil.showUtilization(all=True)
                    
                str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
                str  = 'epoch: {:>d} --- '.format(epoch)
                str  = str + ' | '.join(str1)

                # if self.gpu == 0:
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
            
            # plt.plot_loss_explicit(losses=self.losses["Gloss"], key="Gloss", outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Dloss"], key="Dloss", outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Gloss_ftm"], key="Gloss_ftm", outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Gloss_ali_X"], key="Gloss_ali_X", outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Gloss_ali_z"], key="Gloss_ali_z", outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_X"], key="Gloss_cycle_X", outf=outf,niter=niter)
            # plt.plot_loss_explicit(losses=self.losses["Gloss_cycle_z"], key="Gloss_cycle_z", outf=outf,niter=niter)

            # tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            #     'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'./network/Fef.pth')
            # tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            #     'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'./network/Gdf.pth')    
            # tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dfnets],
            #     'optimizer_state_dict':self.oDfxz.state_dict(),'loss':self.losses},'./network/DsXz.pth')
    
#     def train_hybrid(self):
#         print('Training on filtered signals ...') 
#         globals().update(self.cv)
#         globals().update(opt.__dict__)
#         error = {}
#         for epoch in range(niter):
#             for b,batch in enumerate(trn_loader):
#                 # Load batch
#                 xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
#                 Xd = Variable(xd_data).to(self.gpu) # BB-signal
#                 Xf = Variable(xf_data).to(self.gpu) # LF-signal
#                 zd = Variable(zd_data).to(self.gpu)
#                 zf = Variable(zf_data).to(self.gpu)
# #               # Train G/D
#                 for _ in range(5):
#                     self.alice_train_hybrid_discriminator_adv_xz(Xd,zd,Xf,zf)
#                 for _ in range(1):
#                     self.alice_train_hybrid_generator_adv_xz(Xd,zd,Xf,zf)

#             str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
#             str = 'epoch: {:>d} --- '.format(epoch)
#             str = str + ' | '.join(str1)
#             print(str)
        
#         tsave({'epoch':niter,'model_state_dict':self.Ghz.state_dict(),
#             'optimizer_state_dict':self.oGhxz.state_dict(),'loss':self.losses},'Ghz.pth')    
#         tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dhnets],
#             'optimizer_state_dict':self.oDhzdzf.state_dict(),'loss':self.losses},'DsXz.pth')

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
        # globals().update(self.cv)
        globals().update(self.opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        # if 'broadband' in t and self.strategy['trplt']['broadband']:
        #     n = self.strategy['broadband']
            
        #     plt.plot_generate_classic('broadband',self.Fed,self.Gdd,self.gpu,vtm,\
        #                               vld_loader,pfx="vld_set_bb",outf=outf)
            
        #     plt.plot_features('broadband',self.Fed,self.Gdd,nzd,self.gpu,vtm,vld_loader,pfx='set_bb',outf=outf)

        if 'broadband' in t and self.strategy['trplt']['broadband']:
            n = self.strategy['broadband']
            Fef = deepcopy(self.Fef)
            Gdf = deepcopy(self.Gdf)

            # pdb.set_trace()

            # plot_signal_and_reconstruction(vld_set = self.vld_loader, 
            #         encoder = Fef, 
            #         decoder = Gdf, 
            #         self.gpu  = self.gpu, 
            #         outf    = outf)

            if None not in n:
                print("Loading models {} {}".format(n[0],n[1]))
                Fef.load_state_dict(tload(n[0])['model_state_dict'])
                Gdf.load_state_dict(tload(n[1])['model_state_dict'])
            
            # plt.plot_generate_classic('broadband',Fef,Gdf,device,vtm,\
                                      # self.vld_loader,pfx="vld_set_fl",outf=outf)
            
            # plt.plot_features('broadband',self.Fef,self.Gdf,nzf,device,vtm,vld_loader,pfx='set_fl',outf=outf)

        # if 'hybrid' in t and self.strategy['trplt']['hybrid']:
        #     n = self.strategy['hybrid']
        #     Fef = deepcopy(self.Fef)
        #     Gdd = deepcopy(self.Gdd)
        #     Ghz = deepcopy(self.Ghz)
        #     if None not in n:
        #         print("Loading models {} {} {}".format(n[0],n[1],n[2]))
        #         Fef.load_state_dict(tload(n[0])['model_state_dict'])
        #         Gdd.load_state_dict(tload(n[1])['model_state_dict'])
        #         Ghz.load_state_dict(tload(n[2])['model_state_dict'])
        #     plt.plot_generate_hybrid(Fef,Gdd,Ghz,device,vtm,\
        #                               trn_loader,pfx="trn_set_hb",outf=outf)
        #     plt.plot_generate_hybrid(Fef,Gdd,Ghz,device,vtm,\
        #                               tst_loader,pfx="tst_set_hb",outf=outf)
        #     plt.plot_generate_hybrid(Fef,Gdd,Ghz,device,vtm,\
        #                               vld_loader,pfx="vld_set_hb",outf=outf)

    def compare(self):
        # globals().update(self.cv)
        globals().update(self.opt.__dict__)
        
        t = [y.lower() for y in list(self.strategy.keys())]
        if 'hybrid' in t and self.strategy['trcmp']['hybrid']:
            n = self.strategy['hybrid']
            if None not in n:
                print("Loading models {} {} {}".format(n[0],n[1],n[2]))
                self.Fef.load_state_dict(tload(n[0])['model_state_dict'])
                self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            plt.plot_compare_ann2bb(self.Fef,self.Gdd,self.Ghz,self.gpu,vtm,\
                                    trn_loader,pfx="trn_set_ann2bb",outf=outf)
    def discriminate(self):
        # globals().update(self.cv)
        globals().update(self.opt.__dict__)
        
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
                    Xd = Variable(xd_data).to(self.gpu) # BB-signal
                    Xf = Variable(xf_data).to(self.gpu) # LF-signal
                    zd = Variable(zd_data).to(self.gpu)
                    zf = Variable(zf_data).to(self.gpu)

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

    def get_Dgrad_norm(self, Xd, Xdr, zd, zdr):
        self.Ddxz.train()
        
        batch_size = Xdr.shape[0]

        # Calculate interpolation
        alpha = torch.rand( size = (batch_size, 1, 1), requires_grad=True)
        alphaX = alpha.expand_as(Xd)
        alphaz = alpha.expand_as(zd)

        interpolated = torch.zeros_like(Xdr, requires_grad=True)

        if torch.cuda.is_available():
            alphaX = alphaX.cuda(ngpu-1)
            alphaz = alphaz.cuda(ngpu-1)

        Xhat = alphaX*Xd+(1-alphaX)*Xdr
        zhat = alphaz*zd+(1-alphaz)*zdr

        if torch.cuda.is_available():
            Xhat = Xhat.cuda()
            zhat = zhat.cuda()

        Xhat.requires_grad_(True)
        zhat.requires_grad_(True)

        prob_interpolated = self.Ddxz.critic(zcat(self.DsXd(Xhat),self.Dszd(zhat)))

        if torch.cuda.is_available():
            grad_outputs = torch.ones(prob_interpolated.size(), requires_grad=True).cuda(ngpu-1)
        else:
            grad_outputs = torch.ones(prob_interpolated.size(), requires_grad=True)

        prob_interpolated.requires_grad_(True)
        interpolated.requires_grad_(True)
        grad_outputs.requires_grad_(True)

        # Calculate gradients of probabilities with respect to examples
        gradXz = torch.autograd.grad(outputs=prob_interpolated,\
            inputs = (Xhat,zhat),\
            grad_outputs = grad_outputs,\
            create_graph = True,\
            retain_graph = True,
            only_inputs  = True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradXz = gradXz.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradXz ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gradients_norm


def get_dataset(dataset, nsy = 64, batch_size = 64, rank = 0, world_size = 1):
    dataset = Toyset(nsy = nsy)
    batch_size = batch_size
    train_part = int(0.80*len(dataset))
    vld_part   = len(dataset) - train_part
    train_set, vld_set = torch.utils.data.random_split(dataset, [train_part,vld_part])

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_set,
    #         num_replicas=world_size,
    #         rank=rank)

    # vld_sampler = torch.utils.data.distributed.DistributedSampler(
    #         vld_set,
    #         num_replicas=world_size,
    #         rank=rank)

    trn_loader = torch.utils.data.DataLoader(dataset=train_set, 
                batch_size =batch_size, 
                shuffle    =False,
                num_workers=0,
                pin_memory =True)

    vld_loader = torch.utils.data.DataLoader(dataset=vld_set, 
                batch_size =batch_size, 
                shuffle    =False,
                num_workers=0,
                pin_memory =True)
    return trn_loader, vld_loader