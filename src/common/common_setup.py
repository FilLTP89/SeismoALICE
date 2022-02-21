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
from torch import cuda as tcuda
from torch import device as tdev
from torch import save as tsave
from torch import load as tload
from torch import DoubleTensor as tFT
from torch import LongTensor as tLT
from torch import manual_seed as mseed
from torch.utils.data import DataLoader as tud_dload
from torch.utils.data import BatchSampler as tud_bsmp
from torch.utils.data import RandomSampler as tud_rsmp
from torch.utils.data import ConcatDataset as tdu_cat
import torch.backends.cudnn as cudnn
from common_model import get_truncated_normal
from database_sae import load_dataset,synth_dataset
from database_sae import STEADdataset,ann2bb_dataset
from database_sae import deepbns_dataset
from database_sae import mdof_dataset
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import json

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actions', default='../actions_bb.txt',help='define actions txt')
    parser.add_argument('--strategy', default='../strategy_bb.txt',help='define strategy txt')
    parser.add_argument('--dataset', default='mdof',help='folder | synth | pth | stead | ann2bb | deepbns | mdof')
    parser.add_argument('--dataloaders', type=int, default=1, help="number of dataloaders for recursive training")
    parser.add_argument('--dataroot', default='D:\\Luca\\Dati\\Filippo_data\\damaged_1_1T',help='Path to dataset')
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
    parser.add_argument('--glr', type=float, default=0.0001, help='AE learning rate, default=0.0001')
    parser.add_argument('--rlr', type=float, default=0.0001, help='GAN learning rate, default=0.00005')
    parser.add_argument('--b1', type=float, default=0.5, help='beta1 for Adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--plot', action='store_true', help="flag for plotting")
    parser.add_argument('--outf', default='./imgs', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--mw',type=float,default=4.5,help='magnitude [Mw]')
    parser.add_argument('--dtm',type=float,default=0.01,help='time-step [s]')
    parser.add_argument('--dep',type=float,default=50.,help='epicentral distance [km]')
    parser.add_argument('--scc',type=int,default=0,help='site-class')
    parser.add_argument('--sst',type=int,default=1,help='site')
    parser.add_argument('--scl',type=int,default=1,help='scale [1]')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--nsy',type=int,default=10,help='number of synthetics [1]')
    parser.add_argument('--save_checkpoint',type=int,default=3500,help='Number of epochs for each checkpoint')
    parser.add_argument('--mdof',type=int,default=3,help='Number of channels of the monitoring ssystem (mdof database only)')
    parser.add_argument('--wdof',type=int,nargs='+',default=[1,2,3],help='Channels used by the monitoring system (mdof database only)')
    parser.add_argument('--tdof',default='A',help='Signal content (e.g. U, V, A) (mdof database only)') # eventually 'nargs='+' if different types of signals (e.g. displacements, velocities, accelerations etc.) are considered
    parser.add_argument('--wtdof',nargs='+',default=[3],help='Specify the connection between wdof and tdof (mdof database only)')
    parser.add_argument('--config',default='./config.txt', help='configuration file')
    parser.set_defaults(stack=False,ftune=False,feat=False,plot=True)
    opt = parser.parse_args()

    opt.world_wize = opt.ngpu*opt.nodes

    # import pdb
    # pdb.set_trace()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

    u'''Set-up GPU and CUDA'''
    torch.set_default_type(torch.float16)
    opt.cuda = True if (tcuda.is_available() and opt.cuda) else False
    opt.nch = 3
    device = tdev("cuda" if opt.cuda else "cpu")
    opt.dev = device
    FloatTensor = tcuda.FloatTensor if opt.cuda else tFT
    LongTensor = tcuda.LongTensor if opt.cuda else tLT
    
    #get number of thread in the environment (CPUs)
    opt.ntask =  torch.get_num_threads()
        
    #get number of GPUs in the system environment
    opt.ngpu = torch.cuda.device_count()
    #to no detray propagation values of the gpus
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

    if tcuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    if opt.dataset == 'synth':
        sd_mw = np.log(1.+(2./opt.mw)**2)
        mu_mw = np.log(opt.mw)-0.5*sd_mw
        l=get_truncated_normal(mean=mu_mw,\
                               sd=sd_mw,\
                               low=np.log(3.5),\
                               upp=np.log(8))
        x = l.ppf(np.random.rand(opt.nsy))
        opt.mw  = np.round(np.exp(x), 1) 
        opt.dep = get_truncated_normal(mean=opt.dep,\
                                       sd=50.,low=10.,upp=200.).ppf(np.random.rand(opt.nsy))
        opt.dep = opt.dep.round(0)
        md = {'mw':opt.mw,'dep':opt.dep,'scc':opt.scc,\
          'sst':opt.sst,'dtm':opt.dtm,'scl':opt.scl,'cutoff':opt.cutoff}
        
        md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float32)
        md['nTn'] = md['vTn'].size
        ths_trn,ths_tst,ths_vld,\
        vtm,fsc = synth_dataset(opt.batchPercent,opt.imageSize,opt.latentSize,\
                                opt.nzd,opt.nzf,md=md,nsy=opt.nsy,device=device)
        md['fsc']=fsc
        opt.ncls = md['fsc']['ncat']
        # Create natural period vector 
        opt.vTn = np.arange(0.0,3.05,0.05,dtype=np.float32)
        opt.nTn = md['vTn'].size


        tsave(ths_trn,'./ths_trn.pth')
        tsave(ths_tst,'./ths_tst.pth')
        tsave(ths_vld,'./ths_vld.pth')
        tsave(vtm,    './vtm.pth')
        with open('md.p', 'wb') as handle:
                pickle.dump(md,handle)
        handle.close()
        with open('opt.p', 'wb') as handle:
            pickle.dump(opt,handle)
        handle.close()

    elif opt.dataset == 'folder':
        src = opt.dataroot.split(',')
        for n in range(len(src)):
            src[n] = opj(src[n],'*.*.*.mseed')
        print('dataroots:')
        print(src)

        inv = opt.inventory.split(',')
        for n in range(len(inv)):
            inv[n] = opj(opt.dataroot,'sxml',inv[n])
        print('inventories:')
        print(inv)

        ths_trn,ths_tst,ths_vld,thf_trn,thf_tst,thf_vld,\
        vtm,fsc = load_dataset(opt.batchPercent,\
                               source=src,inventory=inv)
            
        md = {'mw':[],'dep':[],'scc':[],\
          'sst':[],'dtm':vtm[1]-vtm[0],\
          'ntm':vtm.size,'scl':[]}
        opt.ncls = 1
        tsave(ths_trn,'./ths_trn.pth')
        tsave(ths_tst,'./ths_tst.pth')
        tsave(ths_vld,'./ths_vld.pth')
        tsave(thf_trn,'./thf_trn.pth')
        tsave(thf_tst,'./thf_tst.pth')
        tsave(thf_vld,'./thf_vld.pth')
        tsave(vtm,    './vtm.pth')
    
    elif '.pth' in opt.dataset:
        
        # ths_trn = tload(opj(opt.dataroot,'ths_trn_'+opt.dataset))
        # ths_tst = tload(opj(opt.dataroot,'ths_tst_'+opt.dataset))
        # ths_vld = tload(opj(opt.dataroot,'ths_vld_'+opt.dataset))
        vtm = tload(opj(opt.dataroot,'vtm.pth'))
        # pickle_off = open(opj(opt.dataroot,"md.p"),"rb")
        # md = pickle.load(pickle_off)
        # pickle_off.close()
        pickle_off = open(opj(opt.dataroot,"opt.p"),"rb")
        optt = pickle.load(pickle_off)
        pickle_off.close()
        
        for k,v in optt.__dict__.items():
            flag=False
            if k in opt.__dict__:
                try:
                    if type(v).__module__=='numpy' and (v!=opt.__dict__[k]).any():
                        flag=True
                except:
                    pass
                try:
                    if any(v!=opt.__dict__[k]):
                        flag=True
                except:
                    pass 
                try:
                    if v!=opt.__dict__[k]:
                        flag=True
                except:
                    pass
                if flag:
                    optt.__dict__[k] = opt.__dict__[k]
                    
        for k,v in opt.__dict__.items():
            flag=False
            if k not in optt.__dict__:    
                flag=True
            if flag:
                optt.__dict__[k] = opt.__dict__[k]
        opt = optt
    
    elif opt.dataset == 'deepbns':
        src = opj(opt.dataroot,"Hybrid_Database.h5")
        print('dataroots:')
        print(src)
        md = {'dtm':9.452707692307693e-06,'cutoff':opt.cutoff,'ntm':opt.imageSize}
        md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float32)
        md['nTn'] = md['vTn'].size
        ths_trn,ths_tst,ths_vld,\
        vtm,fsc = deepbns_dataset(src,opt.batchPercent,opt.imageSize,opt.latentSize,\
                                  opt.nzd,opt.nzf,md=md,nsy=opt.nsy,device=device)
        md['fsc']=fsc
        opt.ncls = md['fsc']['ncat']
        # Create natural period vector 
        opt.vTn = np.arange(0.0,3.05,0.05,dtype=np.float32)
        opt.nTn = md['vTn'].size
        tsave(ths_trn,'./ths_trn.pth')
        tsave(ths_tst,'./ths_tst.pth')
        tsave(ths_vld,'./ths_vld.pth')
        tsave(vtm,    './vtm.pth')
        with open('md.p', 'wb') as handle:
                pickle.dump(md,handle)
        handle.close()
        with open('opt.p', 'wb') as handle:
                pickle.dump(opt,handle)
        handle.close()

    elif opt.dataset == 'stead':
                
        src = opt.dataroot

        print('dataroots:')
        print(src)
        
        md = {'dtm':0.01,'cutoff':opt.cutoff,'ntm':opt.imageSize}
        md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float32)
        md['nTn'] = md['vTn'].size
        ths_trn,ths_tst,ths_vld,vtm,fsc = STEADdataset(src,
            opt.batchPercent,opt.imageSize,opt.latentSize,\
            opt.nzd,opt.nzf,md=md,nsy=opt.nsy,device=device)
        md['fsc']=fsc
        opt.ncls = md['fsc']['ncat']
        # Create natural period vector 
        opt.vTn = np.arange(0.0,3.05,0.05,dtype=np.float32)
        opt.nTn = md['vTn'].size
        extension= "_nt{0}_ls{1}_nzf{2}_nzd{3}.pth".format(opt.imageSize,opt.latentSize,opt.nzf,opt.nzd)
        tsave(ths_trn,'./ths_trn'+extension)
        tsave(ths_tst,'./ths_tst'+extension)
        tsave(ths_vld,'./ths_vld'+extension)
        tsave(vtm,    './vtm'+extension)
        with open('md.p', 'wb') as handle:
                pickle.dump(md,handle)
        handle.close()
        with open('opt.p', 'wb') as handle:
                pickle.dump(opt,handle)
        handle.close()
        
    elif opt.dataset == 'ann2bb':
        src = opt.dataroot
        print('dataroots:')
        print(src)
        md = {'dtm':0.005,'cutoff':opt.cutoff,'ntm':opt.imageSize}
        md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float32)
        md['nTn'] = md['vTn'].size
        ths_trn,ths_tst,ths_vld,vtm,fsc,md = ann2bb_dataset(src,opt.batchPercent,opt.imageSize,opt.latentSize,\
                                    opt.nzd,opt.nzf,md=md,nsy=opt.nsy,device=device)
        md['fsc']=fsc
        opt.ncls = md['fsc']['ncat']
        # Create natural period vector 
        opt.vTn = np.arange(0.0,3.05,0.05,dtype=np.float32)
        opt.nTn = md['vTn'].size
        tsave(ths_trn,'./ths_trn.pth')
        tsave(ths_tst,'./ths_tst.pth')
        tsave(ths_vld,'./ths_vld.pth')
        tsave(vtm,    './vtm.pth')
        with open('md.p', 'wb') as handle:
                pickle.dump(md,handle)
        handle.close()
        with open('opt.p', 'wb') as handle:
                pickle.dump(opt,handle)
        handle.close()

    elif opt.dataset == 'mdof':
        src = opt.dataroot
        print('dataroots:')
        print(src)
        md = {'dtm':0.01,'cutoff':opt.cutoff,'ntm':opt.imageSize}
        md['vTn'] = np.arange(0.0,3.05,0.05,dtype=np.float32)
        md['nTn'] = md['vTn'].size
        
        ths_trn,ths_tst,ths_vld,vtm,fsc = mdof_dataset(src,opt.batchPercent,opt.imageSize,opt.latentSize,\
                                  opt.nzd,opt.nzf,opt.imageSize,opt.mdof,opt.wdof,opt.tdof,\
                                  opt.wtdof,md=md,nsy=opt.nsy,device=device)
        md['fsc']=fsc
        opt.ncls = md['fsc']['ncat']
        # Create natural period vector 
        opt.vTn = np.arange(0.0,3.05,0.05,dtype=np.float32)
        opt.nTn = md['vTn'].size
        tsave(ths_trn,'./ths_trn.pth')
        tsave(ths_tst,'./ths_tst.pth')
        tsave(ths_vld,'./ths_vld.pth')
        tsave(vtm,    './vtm.pth')
        with open('md.p', 'wb') as handle:
                pickle.dump(md,handle)
        handle.close()
        with open('opt.p', 'wb') as handle:
                pickle.dump(opt,handle)
        handle.close()

    params = {'batch_size': opt.batchSize,\
              'shuffle': True,'num_workers':int(opt.workers)}
    
    # trn_loader,tst_loader,vld_loader = \
    #     dataset2loader(ths_trn,ths_tst,ths_vld,**params)  
#         dataset2loader(ths_trn,thf_trn,wnz_trn,\
#                        ths_tst,thf_tst,wnz_tst,\
#                        ths_vld,thf_vld,wnz_vld,\
#                        **params)   
    actions = pd.read_csv(filepath_or_buffer=opt.actions,sep=',',
                          true_values='True',false_values='False')
    tract = {'tract':actions['tract'].to_dict()}
    trplt = {'trplt':actions['trplt'].to_dict()}
    trcmp = {'trcmp':actions['trcmp'].to_dict()}
    trdis = {'trdis':actions['trdis'].to_dict()}
    strategy = pd.read_csv(filepath_or_buffer=opt.strategy,sep=',',na_values='None')
    strategy = strategy.where((pd.notnull(strategy)), None)
    strategy['strategy'] = strategy.values.tolist()
    strategy = dict(strategy['strategy'].to_dict(),**tract,**trplt,**trcmp,**trdis)
    opt.strategy = strategy
    
    cv = {'parser':parser,'opt':opt,'vtm':vtm,\
          # 'trn_loader':trn_loader,\
          # 'tst_loader':tst_loader,\
          # 'vld_loader':vld_loader,\
          'params':params,\
          'device':device,\
          'FloatTensor':FloatTensor,\
          'LongTensor':LongTensor,\
          # 'md':md
          }
    return cv

def cleanup():
    dist.destroy_process_group()

from torch.utils.data import Sampler
import torch
from torch._six import int_classes as _int_classes

class RandomSamplerNew(Sampler):

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        
        return iter(torch.randperm(n/2).tolist())

    def __len__(self):
        return len(self.data_source)

class BatchSamplerNew(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        batch1 = []
        for idx in self.sampler:
            batch.append(idx)
            batch1.append(idx+max(idx)+1)
            if len(batch) == self.batch_size:
                yield (batch,batch1)
                batch = []
                batch1 = []
        if len(batch) > 0 and not self.drop_last:
            yield (batch,batch1)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

tud_bsmp = BatchSamplerNew
tud_rsmp = RandomSamplerNew

# def dataset2loader(ths_trn,thf_trn,wnz_trn,\
#                    ths_tst,thf_tst,wnz_tst,\
#                    ths_vld,thf_vld,wnz_vld,\
#                    **params):
def dataset2loader(ths_trn,ths_tst,ths_vld,**params):
    
    assert ths_trn
#     assert thf_trn
#     assert wnz_trn
    assert ths_tst
#     assert thf_tst
#     assert wnz_tst
    assert ths_vld
#     assert thf_vld
#     assert wnz_vld
    
    ths_trn = tud_dload(ths_trn,**params)
    ths_tst = tud_dload(ths_tst,**params)
    ths_vld = tud_dload(ths_vld,**params)
#     #
#     thf_trn = tud_dload(thf_trn,**params)
#     thf_tst = tud_dload(thf_tst,**params)
#     thf_vld = tud_dload(thf_vld,**params)
#     #
#     wnz_trn = tud_dload(wnz_trn,**params)
#     wnz_tst = tud_dload(wnz_tst,**params)
#     wnz_vld = tud_dload(wnz_vld,**params)
    
#     bs = params['batch_size']
#     del params['batch_size']
#     
#     ths_trn = tdu_cat((ths_trn,thf_trn))
#     ths_tst = tdu_cat((ths_tst,thf_tst))
#     ths_vld = tdu_cat((ths_vld,thf_vld))
#     
#     trn_bsmp = tud_bsmp(tud_rsmp(ths_trn),bs,drop_last=False)
#     tst_bsmp = tud_bsmp(tud_rsmp(ths_tst),bs,drop_last=False)
#     vld_bsmp = tud_bsmp(tud_rsmp(ths_vld),bs,drop_last=False)
#     
#     trn_params = dict({'batch_sampler':trn_bsmp}.items()+params.items())
#     tst_params = dict({'batch_sampler':tst_bsmp}.items()+params.items())
#     vld_params = dict({'batch_sampler':vld_bsmp}.items()+params.items())
#     
#     ths_trn = tud_dload(ths_trn,**trn_params)
#     ths_tst = tud_dload(ths_tst,**tst_params)
#     ths_vld = tud_dload(ths_vld,**vld_params)
#     
#     trn_params = dict({'batch_sampler':trn_bsmp}.items()+params.items())
#     tst_params = dict({'batch_sampler':tst_bsmp}.items()+params.items())
#     vld_params = dict({'batch_sampler':vld_bsmp}.items()+params.items())
#     
#     thf_trn = tud_dload(thf_trn,**trn_params)
#     thf_tst = tud_dload(thf_tst,**tst_params)
#     thf_vld = tud_dload(thf_vld,**vld_params)
    
#     return (ths_trn,thf_trn,wnz_trn),(ths_tst,thf_tst,wnz_tst),(ths_vld,thf_vld,wnz_vld)
    return ths_trn,ths_tst,ths_vld

if __name__=="__main__":
    cv = setup()
