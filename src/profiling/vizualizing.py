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
from database.database_sae import random_split 
from database.database_sae import thsTensorData
import torch.nn as nn 
import json
import pdb
from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
import time
import GPUtil
from database.toyset import Toyset, get_dataset
from configuration import app
from tqdm import  tqdm,trange

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
    def __init__(self,cv):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        super(trainer, self).__init__()

        self.cv = cv
        self.gr_norm = []

        globals().update(cv)
        globals().update(opt.__dict__)
        b1              = 0.5
        b2              = 0.9999
        self.strategy   = strategy
        self.opt        = opt
        self.start_epoch= 0  

        nzd = opt.nzd
        ndf = opt.ndf

        # torch.backends.cudnn.deterministic  = True
        # torch.backends.cudnn.benchmark      = False

        self.Dnets      = []
        self.optz       = []
        self.oGyx       = None
        # self.dp_mode    = True
        self.losses     = {
            'Dloss':[0],
            'Dloss_ali':[0],
            'Dloss_rec':[0],
            'Dloss_rec_y':[0],
            'Dloss_rec_zy':[0],
            'Gloss':[0],
            'Gloss_cycle_consistency':[0],
            'Gloss_cycle_consistency_y':[0],
            'Gloss_cycle_consistency_zd':[0],
            'Gloss_identity':[0],
            'Gloss_identity_y':[0],
            'Gloss_identity_zd':[0],
            'Gloss_ali':[0]
        }

        self.writer_debug_encoder = SummaryWriter('runs_both/profiling/debug/encoder')
        self.writer_debug_decoder = SummaryWriter('runs_both/profiling/debug/decoder')

        flagT=False
        flagF=False
        t = [y.lower() for y in list(self.strategy.keys())] 

        net = Network(DataParalleleFactory())

        self.trn_loader, self.vld_loader = trn_loader, vld_loader
        self.style='ALICE'
        # act = acts[self.style]
        n = self.strategy['unique']

        self.F_  = net.Encoder(opt.config['F'],  opt)
        self.Gy  = net.Decoder(opt.config['Gy'], opt)
        self.F_  = nn.DataParallel(self.F_).cuda()
        self.Gy  = nn.DataParallel(self.Gy).cuda()

        checkpoint          = tload(n[0])
        self.start_epoch    = checkpoint['epoch']
        self.losses         = checkpoint['loss']
        self.F_.load_state_dict(tload(n[0])['model_state_dict'])
        self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
        self.Gx.load_state_dict(tload(n[2])['model_state_dict'])  

        self.writer_debug_encoder.add_graph(next(iter(self.F_.children())),torch.randn(128,6,4096).cuda())
        self.writer_debug_decoder.add_graph(next(iter(self.Gy.children())), torch.randn(128,512,256).cuda())
        self.bce_loss = BCE(reduction='mean')
        print("Parameters of  Decoders/Decoders ")
        count_parameters(self.FGf)
        print("Parameters of Discriminators ")
        count_parameters(self.Dnets)

    def get_encoder_parameter(self,model):
        cnt = 0
        for module_group in model.children():
            for module in module_group.children():
                for name, params in module.named_parameters():

                    if name in ['Conv1d','ConvTranspose1d']:
                        cnt+=1
                        model_weights.append(params.weight)
                    # writer.add_histogram("layer"+cnt+"/"+name, params)