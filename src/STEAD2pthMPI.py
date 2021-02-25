# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"
r"""Train and Test AAE"""
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
from os.path import join as opj
import numpy as np
import random
import pickle 
from torch import device as tdev
from torch import save as tsave
from torch import load as tload
from torch import FloatTensor as tFT
from torch import LongTensor as tLT
from torch import manual_seed as mseed
from torch.utils.data import DataLoader as tud_dload
from torch.utils.data import BatchSampler as tud_bsmp
from torch.utils.data import RandomSampler as tud_rsmp
from torch.utils.data import ConcatDataset as tdu_cat
import torch.backends.cudnn as cudnn
from common_model import get_truncated_normal
from database_sae import STEADdatasetMPI
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json
from torch.utils.data import Sampler
import torch
from torch._six import int_classes as _int_classes


def to_pth(comm,size,rank):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mdof',help='folder | synth | pth | stead | ann2bb | deepbns | mdof')  #nt4096_ls128_nzf8_nzd32.pth
    parser.add_argument('--dataroot', default='D:\\Luca\\Dati\\Filippo_data\\damaged_1_1T',help='Path to dataset') # '/home/filippo/Data/Filippo/aeolus/ann2bb_as4_') # '/home/filippo/Data/Filippo/aeolus/STEAD/waveforms_11_13_19.hdf5',help='path to dataset') # './database/stead'
    parser.add_argument('--inventory',default='RM07.xml,LXRA.xml,SRN.xml',help='inventories')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
    parser.add_argument('--batchPercent', type=int,nargs='+', default=[0.8,0.1,0.1], help='train/test/validation %')
    parser.add_argument('--niter', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--imageSize', type=int, default=4096, help='the height / width of the input image to network')
    parser.add_argument('--latentSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--cutoff', type=float, default=1., help='cutoff frequency')
    parser.add_argument('--nzd', type=int, default=32, help='size of the latent space')
    parser.add_argument('--nzf', type=int, default=32, help='size of the latent space')
    parser.add_argument('--ngf', type=int, default=32,help='size of G input layer')
    parser.add_argument('--ndf', type=int, default=32,help='size of D input layer')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--plot', action='store_true', help="flag for plotting")
    parser.add_argument('--outf', default='./database/STEADpth', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--mw',type=float,default=4.5,help='magnitude [Mw]')
    parser.add_argument('--dtm',type=float,default=0.01,help='time-step [s]')
    parser.add_argument('--dep',type=float,default=50.,help='epicentral distance [km]')
    parser.add_argument('--scc',type=int,default=0,help='site-class')
    parser.add_argument('--sst',type=int,default=1,help='site')
    parser.add_argument('--scl',type=int,default=1,help='scale [1]')
    parser.add_argument('--nsy',type=int,default=10,help='number of synthetics [1]')
    parser.add_argument('--save_checkpoint',type=int,default=3500,help='Number of epochs for each checkpoint')
    parser.add_argument('--mdof',type=int,default=3,help='Number of channels of the monitoring ssystem (mdof database only)')
    parser.add_argument('--wdof',type=int,nargs='+',default=[1,2,3],help='Channels used by the monitoring system (mdof database only)')
    parser.add_argument('--tdof',default='A',help='Signal content (e.g. U, V, A) (mdof database only)') # eventually 'nargs='+' if different types of signals (e.g. displacements, velocities, accelerations etc.) are considered
    parser.add_argument('--wtdof',nargs='+',default=[3],help='Specify the connection between wdof and tdof (mdof database only)')
    parser.add_argument('--config',default='./config.txt', help='configuration file')
    parser.set_defaults(stack=False,ftune=False,feat=False,plot=True)
    opt = parser.parse_args()

    opt.nch = 3
    device = tdev("cuda" if opt.cuda else "cpu")
    opt.dev = device
    opt.ntask =  torch.get_num_threads()
    opt.ngpu = torch.cuda.device_count()
    ngpu = opt.ngpu
    
    # Try to make an output directory if the latter does not exist
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    # Try to open the .json configuration of the case study
    try:
       with open(opt.config) as json_file:
         opt.config = json.load(json_file)
    except OSError:
        print("|file {}.json not found".format(opt.config))
        opt.config =  None
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    mseed(opt.manualSeed)
    cudnn.benchmark = True
    print("opt.dataset",opt.dataset)

    size = 1
    
    src = opt.dataroot
    print('dataroots:')
    print(src)
    md = {'dtm':0.01,'cutoff':opt.cutoff,'ntm':opt.imageSize}
    md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float64)
    md['nTn'] = md['vTn'].size
    out = STEADdatasetMPI(comm,size,rank,src,
        opt.batchPercent,opt.workers,opt.imageSize,opt.latentSize,\
        opt.nzd,opt.nzf,md=md,nsy=opt.nsy,device=device)
        
    (ths_trn,ths_tst,ths_vld,vtm,fsc) = out
    
    md['fsc']=fsc
    opt.ncls = md['fsc']['ncat']
    # Create natural period vector 
    opt.vTn = np.arange(0.0,3.05,0.05,dtype=np.float64)
    opt.nTn = md['vTn'].size
    tsave(ths_trn,opj(opt.outf,'ths_trn_{:>d}.pth'.format(rank)))
    tsave(ths_tst,opj(opt.outf,'ths_tst_{:>d}.pth'.format(rank)))
    tsave(ths_vld,opj(opt.outf,'ths_vld_{:>d}.pth'.format(rank)))
    tsave(vtm,    opj(opt.outf,'vtm.pth'))
    with open(opj(opt.outf,'md_{:>d}.p'.format(rank)), 'wb') as handle:
            pickle.dump(md,handle)
    handle.close()
    with open(opj(opt.outf,'opt_{:>d}.p'.format(rank)), 'wb') as handle:
            pickle.dump(opt,handle)
    handle.close()