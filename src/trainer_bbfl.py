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
# from dcgaae_model import Encoder, Decoder
# from dcgaae_model import DCGAN_Dx, DCGAN_Dz
from dcgaae_model import *
# from dcgaae_model import DenseEncoder
import plot_tools as plt
from generate_noise import latent_resampling, noise_generator
from generate_noise import lowpass_biquad
from database_sae import random_split 
from leave_p_out import k_folds
from common_setup import dataset2loader
from database_sae import thsTensorData
import json
import pprint as pp
import pdb
from conv_factory import *
# import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP

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
                            + Fed is the Encoder of Broadband signal, herein broadband 
                            signal is named Xd
                            + Gdd is the Decoder(generator) Broadband, herein encoded 
                            signal form broad band is named zd 
                            + Gdf is the Decoder(generator) of the filtred signal, herein 
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
acts={}
acts['ALICE'] = {'Fed' :[LeakyReLU(1.0,inplace=True) for t in range(nly)]+[LeakyReLU(1.0,inplace=True)],
                 'Gdd' :[ReLU(inplace=True) for t in range(nly-1)]+[Tanh()],
                 'Fef' :[LeakyReLU(1.0,inplace=True) for t in range(4)]+[LeakyReLU(1.0,inplace=True)],
                 'Gdf' :[ReLU(inplace=True) for t in range(4)]+[Tanh()],
                 'Ghz' :[ReLU(inplace=True) for t in range(2)]+[LeakyReLU(1.0,inplace=True)],
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
                 'Dsx' :[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 'Dsz' :[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 'Drx' :[LeakyReLU(1.0,inplace=True) for t in range(2)],
                 'Drz' :[LeakyReLU(1.0,inplace=True) for t in range(2)],
                 'Ddxz':[LeakyReLU(1.0,inplace=True) for t in range(2)],
                 'DhXd':[LeakyReLU(1.0,inplace=True) for t in range(3)]}

nlayers = {'Fed':5,'Gdd':5,
           'Fef':5,'Gdf':5,
           'Ghz':3,
           }
kernels = {'Fed':4,'Gdd':4,
           'Fef':4,'Gdf':4,
           'Ghz':3,
           }
strides = {'Fed':4,'Gdd':4,
           'Fef':4,'Gdf':4,
           'Ghz':1,
           }
padding = {'Fed':0,'Gdd':0,
           'Fef':0,'Gdf':0,
           'Ghz':1,
           }
outpads = {'Gdd':0,
           'Gdf':0,
           'Ghz':1,
           }

import subprocess
import os
# def get_gpu_memory_map():
#     """Get the current gpu usage.

#     Returns
#     -------
#     usage: dict
#         Keys are device ids as integers.
#         Values are memory usage as integers in MB.
#     """
#     result = subprocess.check_output(
#         [
#             'nvidia-smi', '--query-gpu=memory.used',
#             '--format=csv,nounits,noheader'
#         ], encoding='utf-8')
#     # Convert lines into a dictionary
#     gpu_memory = [int(x) for x in result.strip().split('\n')]
#     gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
#     return gpu_memory_map



class trainer(object):
    '''Initialize neural network'''
    @profile
    def __init__(self,cv):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        super(trainer, self).__init__()
    
        self.cv = cv
        # define as global variable the cv object. 
        # And therefore this latter is become accessible to the methods in this class
        globals().update(cv)
        # define as global opt and passing it as a dictonnary here
        globals().update(opt.__dict__)
        #import pdb
        #pdb.set_trace()
        #print(opt.config['decoder'])
        # passing the content of file ./strategy_bb_*.txt
        self.strategy=strategy
        self.ngpu = ngpu

        nzd =  opt.nzd
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
        glr = 0.01
        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        """
            This part is for training with the broadband signal
        """
        #we determine in which kind of environnement we are 
        if(opt.ntask ==1 and opt.ngpu >=1):
            print('ModelParallele to Build')
            factory = ModelParalleleFactory()
        elif(opt.ntask >1 and opt.ngpu >=1):
            print('DataParallele to Build')
            factory = DataParalleleFactory()
        else:
            print('environ not found')
        net = Network(factory)
        encoder = net.Encoder(opt.config["encoder"], opt)
        decoder = net.Decoder(opt.config["decoder"], opt)
        DCGAN_Dx, DCGAN_Dz, DCGAN_DXZ = net.Discriminator(opt.config['DsXd'], opt.config['Dszd'], opt.config['Ddxz'], opt)
        
        import pdb
        pdb.set_trace()

        if 'broadband' in t:
            self.style='ALICE'
            act = acts[self.style]
            flagT = True
            # if opt.config exist then change default value defined above 
            if opt.config: 
                nlayers['Fed'] = opt.config['encoder']['nlayers']
                kernels['Fed'] = opt.config['encoder']['kernel']
                strides['Fed'] = opt.config['encoder']['strides']
                padding['Fed'] = opt.config['encoder']['padding']
                nlayers['Gdd'] = opt.config['decoder']['nlayers']
                kernels['Gdd'] = opt.config['decoder']['kernel']
                strides['Gdd'] = opt.config['decoder']['strides']
                padding['Gdd'] = opt.config['decoder']['padding']
                outpads['Gdd'] = opt.config['decoder']['outpads']
            else:
                print('!!! warnings no configuration file found for the broadband\n\tassume default parameters of the programm')
            # read specific strategy for the broadband signal
            n = self.strategy['broadband']
            print("Loading broadband generators")
            
            # # Encoder broadband Fed
            self.Fed = encoder
            # self.Fed = Encoder(ngpu=opt.ngpu,dev=device,nz=nzd,\
            #                    nch=2*nch_tot,ndf=ndf,\
            #                    nly=nlayers['Fed'],ker=kernels['Fed'],\
            #                    std=strides['Fed'],pad=padding['Fed'],\
            #                    dil=1,grp=1,dpc=0.0,act=act['Fed']).to(torch.float32)

            # # Decoder broadband Gdd
            self.Gdd = decoder
            # self.Gdd = Decoder(ngpu=opt.ngpu,nz=2*nzd,nch=nch_tot,\
            #                    ndf=int(ndf//(2**(5-nlayers['Gdd']))),
            #                    nly=nlayers['Gdd'],ker=kernels['Gdd'],
            #                    std=strides['Gdd'],pad=padding['Gdd'],\
            #                    opd=outpads['Gdd'],dpc=0.0,act=act['Gdd']).to(torch.float32)
            # print("|total_memory [GB]:",int(torch.cuda.get_device_properties(device).total_memory//(10**9)))
            
            
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

                self.Dszd = DCGAN_Dz
                self.DsXd = DCGAN_Dx
                self.Ddxz = DCGAN_DXZ
                # if opt.config['Dszd'] :
                #     self.Dszd = DCGAN_Dz(ngpu=opt.ngpu,nz=nzd,ncl=512,n_extra_layers=opt.config['Dszd']['nlayers'],dpc=0.25,
                #                      bn=False,activation=act['Dsz'], opt=opt.config['Dszd']).to(torch.float32)
                # else :
                #     self.Dszd = DCGAN_Dz(ngpu=opt.ngpu,nz=nzd,ncl=512,n_extra_layers=1,dpc=0.25,
                #                      bn=False,activation=act['Dsz']).to(torch.float32)
                #     print('!!! warnings no discriminator configaration for DCGAN_Dz\n\tassume n_extra_layers = 1')
                
                # if opt.config['DsXd'] :
                #     self.DsXd = DCGAN_Dx(ngpu=opt.ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
                #                      n_extra_layers=opt.config['DsXd']['nlayers'],dpc=0.25,activation=act['Dsx'], opt=opt.config['DsXd']).to(torch.float32)
                # else:    
                #     # create a conv1D of 4 layers for the discriminator to transform from (,,3) to (,,512) 
                #     self.DsXd = DCGAN_Dx(ngpu=opt.ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
                #                      n_extra_layers=1,dpc=0.25,activation=act['Dsx']).to(torch.float32)
                #     print('!!!! warnings no discriminator configaration found for DCGAN_Dx\n\tassume n_extra_layers` = 1')
                # # pdb.set_trace()
                
                # if opt.config['Ddxz']:
                #     self.Ddxz = DCGAN_DXZ(ngpu=opt.ngpu,nc=1024,n_extra_layers=opt.config['Ddxz']['nlayers'],dpc=0.25,
                #                       activation=act['Ddxz'], opt=opt.config['Ddxz']).to(torch.float32)
                # else:                 
                #     # create a conv1D of  layers for the discriminator to transform from (*,*,1024) to (*,*,1)
                #     self.Ddxz = DCGAN_DXZ(ngpu=opt.ngpu,nc=1024,n_extra_layers=2,dpc=0.25,
                #                           activation=act['Ddxz']).to(torch.float32)
                #     print('!!! warnings no discriminator configaration for DCGAN_DXZ\n\tassume n_extra_layers = 2')

                # pdb.set_trace()
                self.Ddnets.append(self.Fed)
                self.Ddnets.append(self.Gdd)
                self.Ddnets.append(self.DsXd)  
                self.Ddnets.append(self.Dszd)
                self.Ddnets.append(self.Ddxz)

                # Adam optimization for Ddnets
                self.oDdxz = reset_net(self.Ddnets,lr=0.0001,optim='Adam')
                #Add the same Adam optimization parameter for optzd
                self.optzd.append(self.oDdxz)   
                
            #if we don't the training with the broadband ...
                # we simply load the Python dictionary object that maps each layer to its parameter tensor. 
                # here it is in te 'model_stat_dict' of the strategy_* file passed to the programm
            else:
                if None not in n:
                    print("Broadband generators - NO TRAIN: {0} - {1}".format(*n))
                    self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                else:
                    flagT=False
        """
            This part is for the trainig with the filtred signal
        """
        if 'filtered' in t:
            if opt.config:

                nlayers['Fef'] = opt.config['encoder']['nlayers']
                kernels['Fef'] = opt.config['encoder']['kernel']
                strides['Fef'] = opt.config['encoder']['strides']
                padding['Fef'] = opt.config['encoder']['padding']
                nlayers['Gdf'] = opt.config['decoder']['nlayers']
                kernels['Gdf'] = opt.config['decoder']['kernel']
                strides['Gdf'] = opt.config['decoder']['strides']
                padding['Gdf'] = opt.config['decoder']['padding']
                outpads['Gdf'] = opt.config['decoder']['outpads']
            else:
                print('!!! warnings no configuration file found\n\t assume default values')

            self.style='ALICE'
            act = acts[self.style]
            flagF = True
            n = self.strategy['filtered']
            print("|Loading filtered generators ...")

            self.Fef = Encoder(ngpu=ngpu,dev=device,nz=nzf,nzcl=0,
                               nch=2*nch_tot,ndf=ndf,szs=md['ntm'],
                               nly=nlayers['Fef'],ker=kernels['Fef'],
                               std=strides['Fef'],pad=padding['Fef'],
                               dil=1,grp=1,dpc=0.0,act=act['Fef'])
            print("|Encoder step passed ...")
            self.Gdf = Decoder(ngpu=ngpu,nz=2*nzf,nch=nch_tot,ndf=ndf,
                               nly=nlayers['Gdf'],ker=kernels['Gdf'],
                               std=strides['Gdf'],pad=padding['Gdf'],\
                               opd=outpads['Gdf'],dpc=0.0,act=act['Gdf'])
            print("|Decoder step passed ...")
            #pdb.set_trace()
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
                # creat an Conv1D of 2 layers for the Discriminator Dszf
                if opt.config['Dszf']:
                    self.Dszf = DCGAN_Dz(ngpu=ngpu,nz=nzf,ncl=2*nzf,n_extra_layers=opt.config['Dszf']['nlayers'],dpc=0.25,bn=False,
                                     activation=act['Dsz'],wf=True, opt=opt.config['Dszf']).to(device)
                else:
                    self.Dszf = DCGAN_Dz(ngpu=ngpu,nz=nzf,ncl=2*nzf,n_extra_layers=2,dpc=0.25,bn=False,
                                     activation=act['Dsz'],wf=True).to(device)
                    print('!!! warnings no discriminator configuration found for DCGAN_Dz\n\tassume n_extra_layers = 2')

                if opt.config['DsXf']:
                    self.DsXf = DCGAN_Dx(ngpu=ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
                                     n_extra_layers=opt.config['DsXf']['nlayers'],dpc=0.25,bn=False,activation=act['Dsx'],
                                     wf=True,opt=opt.config['DsXf']).to(device)
                else:
                    self.DsXf = DCGAN_Dx(ngpu=ngpu,isize=256,nc=nch_tot,ncl=512,ndf=64,fpd=1,
                                     n_extra_layers=0,dpc=0.25,bn=False,activation=act['Dsx'],
                                     wf=True).to(device)
                    print('!!! warnings no discriminator configuration found for DCGAN_Dx\n\tassume n_extra_layers = 0')

                if opt.config['Dfxz']:
                    self.Dfxz = DCGAN_DXZ(ngpu=ngpu,nc=512+2*nzf,n_extra_layers=opt.config['Dfxz']['nlayers'],dpc=0.25,
                                      activation=act['Ddxz'],bn=False,wf=True,opt=opt.config['Dfxz']).to(device)
                else:
                    self.Dfxz = DCGAN_DXZ(ngpu=ngpu,nc=512+2*nzf,n_extra_layers=2,dpc=0.25,bn=False,
                                      activation=act['Ddxz'],wf=True).to(device)
                    print('!!! warnings no discriminator configuration found for DCGAN_DXZ\n\t assume n_extra_layers = 2')

                self.Dfnets.append(self.DsXf)
                self.Dfnets.append(self.Dszf)
                self.Dfnets.append(self.Dfxz)
                # recontructionc
                if opt.config['Dsrzf']:
                    self.Dsrzf = DCGAN_Dz(ngpu=ngpu,nz=2*nzf,ncl=2*nzf,n_extra_layers=opt.config['Dsrzf']['nlayers'],dpc=0.25,
                                      bn=False,activation=act['Drz'],opt=opt.config['Dsrzf']).to(device)
                else:
                    self.Dsrzf = DCGAN_Dz(ngpu=ngpu,nz=2*nzf,ncl=2*nzf,n_extra_layers=1,dpc=0.25,
                                      bn=False,activation=act['Drz']).to(device)
                    print('!!! warnings no discriminator configuration found for DCGAN_Dz')

                if opt.config['DsrXf']:
                    self.DsrXf = DCGAN_Dx(ngpu=ngpu,isize=256,nc=2*nch_tot,ncl=512,ndf=64,fpd=1,
                                      n_extra_layers=opt.config['DsrXf']['nlayers'],dpc=0.25,activation=act['Drx'],bn=False,opt=opt.config['DsrXf']).to(device)
                else:
                    self.DsrXf = DCGAN_Dx(ngpu=ngpu,isize=256,nc=2*nch_tot,ncl=512,ndf=64,fpd=1,
                                      n_extra_layers=0,dpc=0.25,bn=False,activation=act['Drx']).to(device)

                self.Dfnets.append(self.DsrXf)
                self.Dfnets.append(self.Dsrzf)
                self.oDfxz = reset_net(self.Dfnets,func=set_weights,lr=rlr,optim='Adam')
                self.optzf.append(self.oDfxz)
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
            act = acts[self.style]
            n = self.strategy['hybrid']
            print("Loading hybrid generators")

            if opt.config['Ghz'] :
                nlayers['Gdf'] = opt.config['decoder']['nlayers']
                kernels['Gdf'] = opt.config['decoder']['kernel']
                strides['Gdf'] = opt.config['decoder']['strides']
                padding['Gdf'] = opt.config['decoder']['padding']
                outpads['Gdf'] = opt.config['decoder']['outpads']
            else:
                print('!!! warnings no config found\n\tassuming default parameters for the hybrid generator')

            self.Ghz = Encoder(ngpu=ngpu,dev=device,nz=nzd,nzcl=0,
                               nch=2*nzf,ndf=nzf*4,szs=32,nly=nlayers['Ghz'],
                               ker=kernels['Ghz'],std=strides['Ghz'],
                               pad=padding['Ghz'],dil=1,grp=1,dpc=0.0,
                               bn=True,act=act['Ghz']).to(device)
            if None not in n:
                print("Hybrid generators - no train: {0} - {1} - {2}".format(*n))
                self.Ghz.load_state_dict(tload(n[2])['model_state_dict'])
            if self.strategy['tract']['hybrid'] or self.strategy['trdis']['hybrid']: 
                if self.style=='WGAN':
                    self.oGhxz = reset_net([self.Ghz],func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    self.oGhxz = reset_net([self.Ghz],func=set_weights,lr=glr,b1=b1,b2=b2)
                self.optzh.append(self.oGhxz)

                if opt.config['Dsrzd']:
                    self.Dsrzd = DCGAN_DXZ(ngpu=ngpu,nc=2*nzd,n_extra_layers=opt.config['Dsrzd']['layers'],dpc=0.25,
                                           activation=act['Ddxz'],wf=False).to(device)
                else:
                    self.Dsrzd = DCGAN_DXZ(ngpu=ngpu,nc=2*nzd,n_extra_layers=2,dpc=0.25,
                                           activation=act['Ddxz'],wf=False).to(device)
                
                if self.style=='WGAN':
                    if opt.config['DsrXd']:
                        self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=3,ker=opt.config['DsrXd']['kernels'],\
                                     std=opt.config['DsrXd']['strides'],\
                                     pad=opt.config['DsrXd']['padding'],\
                                     dil=opt.config['DsrXd']['dilation'],\
                                     grp=1,dpc=0.25,bn=False,\
                                     act=act['DhXd']).to(device)
                    else:
                        self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                         ndf=ndf,szs=md['ntm'],nly=3,ker=3,std=2,\
                                         pad=1,dil=1,grp=1,dpc=0.25,bn=False,\
                                         act=act['DhXd']).to(device)
                        print('!!! warnings no configuration found for DsrXd')
                    self.Dhnets.append(self.Dsrzd)
                    self.Dhnets.append(self.DsrXd)
                    self.oDhzdzf = reset_net(self.Dhnets,func=set_weights,lr=rlr,optim='rmsprop')
                else:
                    if opt.config['DsrXd']:
                        self.DsrXd = Encoder(ngpu=ngpu,dev=device,nz=1,nzcl=0,nch=2*nch_tot,
                                     ndf=ndf,szs=md['ntm'],nly=op.config['DsrXd']['nlayers'],\
                                     ker=opt.config['DsrXd']['kernel'],\
                                     std=opt.config['DsrXd']['strides'],\
                                     pad=opt.config['DsrXd']['padding'],\
                                     dil=opt.config['DsrXd']['dilation'],\
                                     grp=1,dpc=0.25,bn=True,\
                                     act=act['DhXd']).to(device)    
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
        
        #end of constructior

    ''' Methode that discriminate real and fake signal for broadband type '''
    @profile
    def discriminate_broadband_xz(self,Xd,Xdr,zd,zdr):
        
        # Discriminate real
        print("|In discriminate_broadband_xz function")
        print("\t||Xd : ",Xd.shape,"\tzdr : " ,zdr.shape)
        print("\t||Xdr : ",Xdr.shape,"\tzd : ", zd.shape)
        # import pdb
        #pdb.set_trace()

        # put the out put on the same GPU
        # pdb.set_trace()
        a = self.DsXd(Xd)
        b = self.Dszd(zdr)
        
        print("\t||DsXd(Xd) : ", a.shape,"\tDszd(zdr) : ",b.shape)
        

        zrc = zcat(a,b)
        print("\t\t|||zrc : ", zrc.shape)
        DXz = self.Ddxz(zrc)
        print("\t\t|||DXz : ", DXz.shape)
        # Discriminate fake
        c = self.DsXd(Xdr)
        d = self.Dszd(zd)
        print("\t||DsXd(Xdr) : ",c.shape,"\tDszd(zd) : ", d.shape)
        zrc = zcat(c,d)
        DzX = self.Ddxz(zrc)
        
        return DXz,DzX
        #end of discriminate_broadband_xz function

    ''' Methode that discriminate real and fake signal for filtred type '''
    @profile
    def discriminate_filtered_xz(self,Xf,Xfr,zf,zfr):
        print("|In discriminate_filtered_xz function ...") 
        # Discriminate real
        print("\t||Xf : ",Xf.shape,"\tzfr : ",zfr.shape)
        #pdb.set_trace()
        ftz = self.Dszf(zfr)
        ftX = self.DsXf(Xf)
        print("\t||ftX[0] : ", ftX[0].shape,"\tftz[0] : ",ftz[0].shape)
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
        print("|[1]In the alice_train_broadband_generator_explicit_xz function  ...") 
        print("\t||Xd:",Xd.shape,"\tzd ",zd.shape)
        # Set-up training
        zerograd(self.optzd)
        self.Fed.eval(),self.Gdd.eval()
        self.DsXd.train(),self.Dszd.train(),self.Ddxz.train()
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        # 1. Concatenate inputs
        X_inp = zcat(Xd,wnx.to(Xd.device))
        z_inp = zcat(zd,wnz.to(zd.device))
        #GPUtil.showUtilization()
        print("\t||X_inp : ",X_inp.shape,"\tz_inp : ",z_inp.shape)

        # 2. Generate conditional samples
        
        X_gen = self.Gdd(z_inp)
        z_gen = self.Fed(X_inp)
        torch.cuda.empty_cache()
        # pdb.set_trace()
        print("\t||X_gen : ",X_gen.shape,"\tz_gen : ",z_gen.shape)
        # z_gen = latent_resampling(self.Fed(X_inp),nzd,wn1)

        # 3. Cross-Discriminate XZ
        Dxz,Dzx = self.discriminate_broadband_xz(Xd,X_gen,zd,z_gen)
        print("\t||After discriminate_broadband_xz") 
        # 4. Compute ALI discriminator loss
        print("\t||Dzx : ", Dzx.shape,"Dxz : ",Dxz.shape)
        Dloss_ali = -torch.mean(ln0c(Dzx)+ln0c(1.0-Dxz))
        
        # Total loss
        #pdb.set_trace()
        Dloss = Dloss_ali.to(3,non_blocking=True)
        Dloss.backward()
        torch.cuda.empty_cache()
        self.oDdxz.step()
        zerograd(self.optzd)
        self.losses['Dloss_t'].append(Dloss.tolist())
        # GPUtil.showUtilization(all=True)
    
    @profile
    def alice_train_broadband_generator_explicit_xz(self,Xd,zd):
        # print("|[2]In alice_train_broadband_generator_explicit_xz ...")
        # Set-up training
        zerograd(self.optzd)
        self.Fed.train(),self.Gdd.train()
        self.DsXd.train(),self.Dszd.train(),self.Ddxz.train()
        #pdb.set_trace()
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        #Put wnx and wnz and the same device of X_inp and z_inp
        wnx = wnx.to(0)
        wnz = wnz.to(0)
        # 1. Concatenate inputs
        X_inp = zcat(Xd,wnx)
        z_inp = zcat(zd,wnz)
        print("\t||X_inp", X_inp.shape,"\t||z_inp",z_inp.shape)
        # 2. Generate conditional samples
        
        X_gen = self.Gdd(z_inp)
        z_gen = self.Fed(X_inp)
        print("\t||X_gen", X_gen.shape,"\t||z_gen",z_gen.shape)
        # z_gen = latent_resampling(self.Fed(X_inp),nzd,wn1)
        
        # 3. Cross-Discriminate XZ
        Dxz,Dzx = self.discriminate_broadband_xz(Xd,X_gen,zd,z_gen)
        print("\t||Dxz", Dxz.shape,"\t||Dzx",Dzx.shape)
        # 4. Compute ALI Generator loss WGAN
        Gloss_ali = torch.mean(-Dxz +Dzx).to(1,non_blocking=True)
        
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,device,rndm_args)
        # # passign values to the CPU 0
        
        wnx = wnx
        wnz = wnz
        wn1 = wn1
        # 1. Concatenate inputs
        X_gen = zcat(X_gen,wnx.to(X_gen.device,non_blocking=True))
        z_gen = zcat(z_gen,wnz.to(z_gen.device,non_blocking=True))
        
        # 2. Generate reconstructions
        X_rec = self.Gdd(z_gen).to(1,non_blocking=True)
        z_rec = self.Fed(X_gen).to(1,non_blocking=True)
        # z_rec = latent_resampling(self.Fed(X_gen),nzd,wn1)
        print("\t||X_rec", X_rec.shape,"\t||z_rec",z_rec.shape)
        # 3. Cross-Discriminate XX
        # pdb.set_trace()
        Gloss_cycle_X = torch.mean(torch.abs(Xd.to(1)-X_rec)).to(1,non_blocking=True)  
        
        # 4. Cross-Discriminate ZZ
        Gloss_cycle_z = torch.mean(torch.abs(zd.to(1)-z_rec)).to(1,non_blocking=True)

        # Total Loss
        Gloss = Gloss_ali + 10. * Gloss_cycle_X + 100. * Gloss_cycle_z

        #torch.cuda.empty_cache()
        Gloss.backward()
        # GPUtil.showUtilization(all=True)
        
        # pp.pprint(dict(torch.cuda.memory_snapshot()), indent=4)
        self.oGdxz.step()
        zerograd(self.optzd)
        
        self.losses['Gloss_t'].append(Gloss.tolist()) 
        self.losses['Gloss_x'].append(Gloss_cycle_X.tolist())
        self.losses['Gloss_z'].append(Gloss_cycle_z.tolist())
        # GPUtil.showUtilization(all=True)
        torch.cuda.empty_cache()
    ####################
    ##### FILTERED #####
    ####################
    def alice_train_filtered_discriminator_adv_xz(self,Xf,zf):
        # Set-up training
        print("|In function alice_train_filtered_discriminator_adv_xzl")
        print("\t||Xf : ", Xf.shape,"\tzf : ",zf.shape)
        zerograd(self.optzf)
        self.Fef.eval(),self.Gdf.eval()
        self.DsXf.train(),self.Dszf.train(),self.Dfxz.train()
        self.DsrXf.train(),self.Dsrzf.train()
         
        # 0. Generate noise
        wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,device,rndm_args)
         
        # 1. Concatenate inputs
        X_inp = zcat(Xf,wnx)
        z_inp = zcat(zf,wnz)
        print("\t||X_inp :", X_inp.shape,"\tz_inp : " ,z_inp.shape)
        # 2. Generate conditional samples
        X_gen = self.Gdf(z_inp).cuda(1)
        z_gen = self.Fef(X_inp).cuda(1)
        # z_gen = latent_resampling(self.Fef(X_inp),nzf,wn1)
        print("\t||X_gen : ", X_gen.shape,"\tz_gen : ",z_gen.shape)
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
        print('Training on broadband signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)   
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                # pdb.set_trace()
                xd_data,_,zd_data,_,_,_,_ = batch
                Xd = Variable(xd_data).to(0,non_blocking=True) # BB-signal
                zd = Variable(zd_data).to(0,non_blocking=True)
                # Train G/D
                for _ in range(5):
                    self.alice_train_broadband_discriminator_explicit_xz(Xd,zd)
                    
                for _ in range(1):
                    self.alice_train_broadband_generator_explicit_xz(Xd,zd)
                   
                
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
            if epoch%save_checkpoint==0:
                tsave({'epoch':epoch,'model_state_dict':self.Fed.state_dict(),
                       'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},
                       'Fed_{}.pth'.format(epoch))
                tsave({'epoch':epoch,'model_state_dict':self.Gdd.state_dict(),
                       'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},
                       'Gdd_{}.pth'.format(epoch))    
                tsave({'model_state_dict':self.Dszd.state_dict(),
                       'optimizer_state_dict':self.oDdxz.state_dict()},'Dszd_bb_{}.pth'.format(epoch))
                tsave({'model_state_dict':self.DsXd.state_dict(),
                       'optimizer_state_dict':self.oDdxz.state_dict()},'DsXd_bb_{}.pth'.format(epoch))    
                tsave({'model_state_dict':self.Ddxz.state_dict(),
                       'optimizer_state_dict':self.oDdxz.state_dict()},'Ddxz_bb_{}.pth'.format(epoch))
        plt.plot_loss_dict(nb=niter,losses=self.losses,title='loss_classic',outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fed.state_dict(),
            'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'Fed.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdd.state_dict(),
            'optimizer_state_dict':self.oGdxz.state_dict(),'loss':self.losses,},'Gdd.pth')    
        tsave({'model_state_dict':self.Dszd.state_dict()},'Dszd_bb.pth')
        tsave({'model_state_dict':self.DsXd.state_dict()},'DsXd_bb.pth')    
        tsave({'model_state_dict':self.Ddxz.state_dict()},'Ddxz_bb.pth')
         
    @profile
    def train_filtered_explicit(self):
        print('Training on filtered signals ...') 
        globals().update(self.cv)
        globals().update(opt.__dict__)
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                _,xf_data,_,zf_data,_,_,_ = batch
                Xf = Variable(xf_data).to(device,non_blocking=True) # LF-signal
                zf = Variable(zf_data).to(device,non_blocking=True)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_filtered_discriminator_explicit_xz(Xf,zf)
                for _ in range(1):
                    self.alice_train_filtered_generator_explicit_xz(Xf,zf)
    
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
            if epoch%save_checkpoint==0:
                tsave({'epoch':epoch,'model_state_dict':self.Fef.state_dict(),
                       'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                       'Fef_{}.pth'.format(epoch))
                tsave({'epoch':epoch,'model_state_dict':self.Gdf.state_dict(),
                       'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                       'Gdf_{}.pth'.format(epoch))    
                tsave({'model_state_dict':self.Dszf.state_dict(),
                       'optimizer_state_dict':self.oDfxz.state_dict()},'Dszd_fl_{}.pth'.format(epoch))
                tsave({'model_state_dict':self.DsXf.state_dict(),
                       'optimizer_state_dict':self.oDfxz.state_dict()},'DsXd_fl_{}.pth'.format(epoch))    
                tsave({'model_state_dict':self.Dfxz.state_dict(),
                       'optimizer_state_dict':self.oDfxz.state_dict()},'Ddxz_fl_{}.pth'.format(epoch))
        #plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_filtered',outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},'Fef.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},'Gdf.pth')    
    @profile
    def train_filtered(self):
        print("[!] In function train_filtred ... ")
        globals().update(self.cv)
        globals().update(opt.__dict__)
	
        for epoch in range(niter):
            for b,batch in enumerate(trn_loader):
                # Load batch
                _,xf_data,_,zf_data,_,_,_ = batch
                Xf = Variable(xf_data).to(device,non_blocking=True) # LF-signal
                zf = Variable(zf_data).to(device,non_blocking=True)
                print("Xf and zf", Xf.shape, zf.shape)
#               # Train G/D
                for _ in range(5):
                    self.alice_train_filtered_discriminator_adv_xz(Xf,zf)
                for _ in range(1):
                    self.alice_train_filtered_generator_adv_xz(Xf,zf)
    
            str1 = ['{}: {:>5.3f}'.format(k,np.mean(np.array(v[-b:-1]))) for k,v in self.losses.items()]
            str = 'epoch: {:>d} --- '.format(epoch)
            str = str + ' | '.join(str1)
            print(str)
            if epoch%save_checkpoint==0:
                tsave({'epoch':epoch,'model_state_dict':self.Fef.state_dict(),
                       'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                       'Fef_{}.pth'.format(epoch))
                tsave({'epoch':epoch,'model_state_dict':self.Gdf.state_dict(),
                       'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses,},
                       'Gdf_{}.pth'.format(epoch))    
                tsave({'model_state_dict':self.Dszf.state_dict(),
                       'optimizer_state_dict':self.oDfxz.state_dict()},'Dszd_fl_{}.pth'.format(epoch))
                tsave({'model_state_dict':self.DsXf.state_dict(),
                       'optimizer_state_dict':self.oDfxz.state_dict()},'DsXd_fl_{}.pth'.format(epoch))    
                tsave({'model_state_dict':self.Dfxz.state_dict(),
                       'optimizer_state_dict':self.oDfxz.state_dict()},'Ddxz_fl_{}.pth'.format(epoch))
        #plt.plot_loss(niter,len(trn_loader),self.losses,title='loss_filtered',outf=outf)
        tsave({'epoch':niter,'model_state_dict':self.Fef.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'Fef.pth')
        tsave({'epoch':niter,'model_state_dict':self.Gdf.state_dict(),
            'optimizer_state_dict':self.oGfxz.state_dict(),'loss':self.losses},'Gdf.pth')    
        tsave({'epoch':niter,'model_state_dict':[Dn.state_dict() for Dn in self.Dfnets],
            'optimizer_state_dict':self.oDfxz.state_dict(),'loss':self.losses},'DsXz.pth')
    
    @profile
    def train_hybrid(self):
        print('Training on filtered signals ...') 
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
            plt.plot_generate_classic('broadband',self.Fed,self.Gdd,device,vtm,\
                                      vld_loader,pfx="vld_set_bb",outf=outf)
            #plt.plot_gofs(tag=['broadband'],Fef=self.Fef,Gdf=self.Gdf,Fed=self.Fed,\
            #        	  Gdd=self.Gdd,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
            #              pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb'},\
            #             outf=outf)
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
            plt.plot_generate_classic('filtered',Fef,Gdf,device,vtm,\
                                      vld_loader,pfx="vld_set_fl",outf=outf)
            #plt.plot_gofs(tag=['filtered'],Fef=self.Fef,Gdf=self.Gdf,Fed=self.Fed,\
            #              Gdd=self.Gdd,Fhz=self.Fhz,Ghz=self.Ghz,dev=device,vtm=vtm,trn_set=trn_loader,\
            #              pfx={'broadband':'set_bb','filtered':'set_fl','hybrid':'set_hb'},\
            #              outf=outf)
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
                #pdb.set_trace()
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
   