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
from plot_tools import plot_eg_pg
from obspy.signal.tf_misfit import plot_tf_gofs, eg,pg
from obspy.signal.tf_misfit import eg, pg
from generate_noise import latent_resampling, noise_generator
from generate_noise import lowpass_biquad
from database_sae import random_split 
from leave_p_out import k_folds
from ex_common_setup import dataset2loader
from database_sae import thsTensorData
import json
from pytorch_summary import summary
import pdb
from conv_factory import *
import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import torch.distributed as dist
from laploss import *
from pymssa.mssa import MSSA

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
        self.data = './database/tweaked/data/latent_sampling.pth'

        # define as global variable the cv object. 
        # And therefore this latter is become accessible to the methods in this class
        globals().update(cv)
        # define as global opt and passing it as a dictonnary here
        globals().update(opt.__dict__)
        self.signals  = self.extraction_data(data = self.data)
        # self.signals  = self.extraction_data()
        # path = './'
        # encoder = T.load_net('./network/trained/nzd32')
        # latent_space = encoder(Xd)

        # @profile
    def generate(self):
        # pdb.set_trace()
        X = np.transpose(self.signals)
        window_size = 32 #128
        n_groups = 32#32
        ssa = MSSA(window_size=window_size,
                 n_components=n_groups,
                 variance_explained_threshold=0.95,
                 pa_percentile_threshold=95,
                 svd_method='randomized',
                 varimax=False,
                 verbose=True)
        # pdb.set_trace()
        ssa.fit(X)
        # w_corr =  ssa.w_correlation(X)
        np.save('M_ssa_example_w{}_gr{}.npy'.format(window_size,n_groups),ssa.components_)
        print('saving M_ssa_example_w{}_gr{}.npy ...'.format(window_size,n_groups))
        #original data
        # red_idx = 2
        # red_data = X

        
        ssa_components = np.load('M_ssa_example_w{}_gr{}.npy'.format(window_size,n_groups))
        idx = ssa_components.shape[0]
        red_data = X
        cumulative_recon = np.zeros_like(red_data[:, 0])
        # pdb.set_trace()
        # w_corr = ssa.w_correlation(ssa.components_)
        EG = []
        PG = []
        dt = 0.10
        for red_idx in range(idx):
            cumulative_recon = np.zeros_like(red_data[:, 0])
            for comp in range(n_groups):  
                fig, ax = plt.subplots(figsize=(18, 7))
                current_component = ssa_components[red_idx, :, comp]
                cumulative_recon = cumulative_recon + current_component
                
                ax.plot(range(red_data.shape[0]), red_data[:, red_idx], lw=3, alpha=0.2, c='k')
                ax.plot(range(red_data.shape[0]), cumulative_recon, lw=3, c='darkgoldenrod', alpha=0.6, label='cumulative'.format(comp))
                ax.plot(range(red_data.shape[0]), current_component, lw=3, c='steelblue', alpha=0.8, label='component={}'.format(comp))
                
                ax.legend()
              
            plt.savefig(os.path.join(outf,"mssa%s_%s.png"%(red_idx,comp)),\
                    bbox_inches='tight',dpi = 300)
            print("mssa%s_%s.png"%(red_idx,comp))
            EG.append(eg(current_component,cumulative_recon,dt=dt,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                    st2_isref=True,a=10.,k=1))
            PG.append(pg(current_component,cumulative_recon,dt=dt,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                    st2_isref=True,a=10.,k=1))         
        # pdb.set_trace()
        plot_eg_pg(EG,PG, outf)
        # w_corr =  ssa.w_correlation(X)  


    def extraction_data(self,data= 'trn_loader'):
        limit = 100
        NS = []
        EW = []
        UD = []
        if data == 'trn_loader':
            for _,batch in enumerate(trn_loader):
                # Load batch
                # pdb.set_trace()
                xd_data,_,_,_,_,_,_ = batch
                Xd = Variable(xd_data)# BB-signal
                # zd = Variable(zd_data).to(ngpu-1)
                NS.append(Xd[:,0,:])
                EW.append(Xd[:,1,:])
                UD.append(Xd[:,2,:])
            # pdb.set_trace()
            NS = torch.cat(NS).numpy()
            EW = torch.cat(EW).numpy()
            UD = torch.cat(UD).numpy()
            signals  = np.vstack((NS[0:limit,:], EW[0:limit,:], UD[0:limit,:]))
        else:
            try:
                # pdb.set_trace()
                signals = torch.load(data)
                z1 = signals[:,0,:].cpu().detach().numpy()
                z2 = signals[:,1,:].cpu().detach().numpy()
                z3 = signals[:,0,:].cpu().detach().numpy()
                signals = np.vstack((z1,z2,z3))
            except Exception as  e:
                print('data not found !!!')
                raise e

        return signals