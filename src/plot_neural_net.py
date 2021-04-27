# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
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
from ex_database_sae import thsTensorData
import json
from pytorch_summary import summary
import pdb
from conv_factory import *
# import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch.distributed as dist

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

import numpy as np
import pandas as pd
import sklearn as skl
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.model_selection import train_test_split
import h5py

import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import CustomJS, Slider, ColumnDataSource
import bokeh.io
from bokeh.io import curdoc
from bokeh.models.renderers import GlyphRenderer

import yaml
import os

from numpy.fft import fft
from numpy.fft import ifft

import time as timer_sec

later = 0.0

import sys
sys.path.append('../')
from PlotNeuralNet.pycore.tikzeng import *



# TO EJECTUTE
# bokeh serve --show .\ssa_app_seismo.py

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

        # passing the content of file ./strategy_bb_*.txt
        self.strategy=strategy
        self.ngpu = ngpu
        # dist.init_process_group("gloo", rank=ngpu, world_size=1)

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
            # self.style='ALICE'
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
                # self.DsXf = net.DCGAN_Dx(opt.config['DsXf'], opt)
                # self.Dszf = net.DCGAN_Dz(opt.config['Dszf'], opt)
                # self.Dfxz = net.DCGAN_DXZ(opt.config['Dfxz'], opt)
            else:
                if None not in n:
                    print("Broadband generators - NO TRAIN: {0} - {1}".format(*n))
                    self.Fed.load_state_dict(tload(n[0])['model_state_dict'])
                    self.Gdd.load_state_dict(tload(n[1])['model_state_dict'])
                else:
                    flagT=False

    def generate(self):
        # defined your arch
        print("reading architecture of network ...")
        model_children = list(self.Fed.children())

        for child in model_children:
            if type(child)==nn.Conv1d:
                # no_of_layers+=1
                # pdb.set_trace()
                conv_layers.append(child)

            elif type(child)==nn.Sequential:
                for layer in child.children():
                    # pdb.set_trace()
                    if type(layer)==nn.Conv1d:
                        # no_of_layers+=1
                        conv_layers.append(layer)

        arch = [
            to_head( '..' ),
            to_cor(),
            to_begin(),
            to_Conv("conv1", 512, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2 ),
            to_Pool("pool1", offset="(0,0,0)", to="(conv1-east)"),
            to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
            to_connection( "pool1", "conv2"),
            to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
            to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
            to_connection("pool2", "soft1"),
            to_end()
        ]

        namefile ="test_network"
        print("generating the file  {}.tex".format(namefile))
        code = to_generate(arch, namefile + '.tex' )
