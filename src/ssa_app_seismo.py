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
from common_setup import dataset2loader
from database_sae import thsTensorData
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

        self.NS = []
        self.EW = []
        self.UD = []
        
        for _,batch in enumerate(trn_loader):
            # Load batch
            # pdb.set_trace()
            xd_data,_,_,_,_,_,_ = batch
            Xd = Variable(xd_data)# BB-signal
            # zd = Variable(zd_data).to(ngpu-1)
            self.NS.append(Xd[:,0,:])
            self.EW.append(Xd[:,1,:])
            self.UD.append(Xd[:,2,:])

        # pdb.set_trace()

        self.NS = torch.cat(self.NS).numpy()
        self.EW = torch.cat(self.EW).numpy()
        self.UD = torch.cat(self.UD).numpy()

        # @profile
    def generate(self):
        X = self.EW[79,:]
        plt.plot(range(0,4096),X)
        plt.savefig(os.path.join(outf,"True_signal_.png"),\
                        bbox_inches='tight',dpi = 300)
        print("saving True_signal_.png")
        plt.close()

        _ssa = SSA(X, 32)
        _ssa.plot_wcorr()
        plt.savefig(os.path.join(outf,"corr_signal_.png"),\
                        bbox_inches='tight',dpi = 300)
        print("saving corr_signal_.png")
        plt.close()

        X_rec = _ssa.reconstruct([0,1,2,3,5,6,10,12])
        X_rec.plot()

        print("erreur : ")
        _ssa.orig_TS.plot(alpha=0.4)
        plt.xlabel("$t$")
        plt.ylabel(r"$\tilde{F}_i(t)$")
        plt.savefig(os.path.join(outf,"rec_signal.png"),\
                        bbox_inches='tight',dpi = 300)
        print("reconstruct signal.png")

        print("precision {}".format((X_rec-X).mean()/X.mean()))

class SSA(object):
    
    __supported_types = (pd.Series, np.ndarray, list)
    
    def __init__(self, tseries, L, save_mem=True):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.
        https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition/notebook#2.-Introducing-the-SSA-Method
        
        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list. 
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.
        
        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """
        
        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")
        
        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")
        
        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1
        
        # Embed the time series in a trajectory matrix
        self.X = np.array([self.orig_TS.values[i:L+i] for i in range(0, self.K)]).T
        
        # Decompose the trajectory matrix
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.d = np.linalg.matrix_rank(self.X)
        
        self.TS_comps = np.zeros((self.N, self.d))
        
        if not save_mem:
            # Construct and save all the elementary matrices
            self.X_elem = np.array([ self.Sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d) ])

            # Diagonally average the elementary matrices, store them as columns in array.           
            for i in range(self.d):
                X_rev = self.X_elem[i, ::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.V = VT.T
        else:
            # Reconstruct the elementary matrices without storing them
            for i in range(self.d):
                X_elem = self.Sigma[i]*np.outer(self.U[:,i], VT[i,:])
                X_rev = X_elem[::-1]
                self.TS_comps[:,i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0]+1, X_rev.shape[1])]
            
            self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."
            
            # The V array may also be very large under these circumstances, so we won't keep it.
            self.V = "Re-run with save_mem=False to retain the V matrix."
        
        # Calculate the w-correlation matrix.
        self.calc_wcorr()
            
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)
            
    
    def reconstruct(self, indices):
        """
        Reconstructs the time series from its elementary components, using the given indices. Returns a Pandas Series
        object with the reconstructed time series.
        
        Parameters
        ----------
        indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
        """
        if isinstance(indices, int): indices = [indices]
        
        ts_vals = self.TS_comps[:,indices].sum(axis=1)
        return pd.Series(ts_vals, index=self.orig_TS.index)
    
    def calc_wcorr(self):
        """
        Calculates the w-correlation matrix for the time series.
        """
             
        # Calculate the weights
        w = np.array(list(np.arange(self.L)+1) + [self.L]*(self.K-self.L-1) + list(np.arange(self.L)+1)[::-1])
        
        def w_inner(F_i, F_j):
            return w.dot(F_i*F_j)
        
        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array([w_inner(self.TS_comps[:,i], self.TS_comps[:,i]) for i in range(self.d)])
        F_wnorms = F_wnorms**-0.5
        
        # Calculate Wcorr.
        self.Wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i+1,self.d):
                self.Wcorr[i,j] = abs(w_inner(self.TS_comps[:,i], self.TS_comps[:,j]) * F_wnorms[i] * F_wnorms[j])
                self.Wcorr[j,i] = self.Wcorr[i,j]
    
    def plot_wcorr(self, min=None, max=None):
        """
        Plots the w-correlation matrix for the decomposed time series.
        """
        if min is None:
            min = 0
        if max is None:
            max = self.d
        
        if self.Wcorr is None:
            self.calc_wcorr()
        
        ax = plt.imshow(self.Wcorr)
        plt.xlabel(r"$\tilde{F}_i$")
        plt.ylabel(r"$\tilde{F}_j$")
        plt.colorbar(ax.colorbar, fraction=0.045)
        ax.colorbar.set_label("$W_{i,j}$")
        plt.clim(0,1)
        
        # For plotting purposes:
        if max == self.d:
            max_rnge = self.d-1
        else:
            max_rnge = max
        
        plt.xlim(min-0.5, max_rnge+0.5)
        plt.ylim(max_rnge+0.5, min-0.5)


