# -*- coding: utf-8 -*-
#!/usr/bin/env python
u"""
Main file train and test the SAE
"""
u'''Required modules'''
import numpy as np
import torch
import h5py
import pandas as pd
import random as rnd
from os.path import join as osj
from torch import from_numpy as np2t
from common_nn import to_categorical
from common_torch import o1l
from torch.utils import data as data_utils
from torch import randperm as rndp
from torch.utils.data.dataset import Subset, _accumulate
from torch.utils.data import DataLoader as dloader
from obspy import Stream, read, read_inventory
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import generate_noise as gng
from generate_noise import filter_ths_tensor as flt, filter_signal
from generate_noise import lowpass_biquad
from generate_noise import filter_Gauss_noise as flg
from generate_noise import Gauss_noise as gn
from generate_noise import pollute_ths_tensor as pgn
from synthetic_generator import sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.integrate import cumtrapz
from torch import FloatTensor as tFT
from ss_process import rsp
from scipy.signal import detrend, windows
from scipy.io import loadmat

rndm_args = {'mean': 0, 'std': 1}
dirs = {'ew':0,'ns':1,'ud':2}
u'''General informations'''

__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

def nxtpow2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def _fit_transform(fsc,x):
    y = fsc.fit_transform(x)
    return y
def _transform(fsc,x):
    y = fsc.transform(x)
    return y

def _standard_scaler():
    print('StandardScaling...')
    fsc = StandardScaler()
    return fsc

def _robust_scaler():
    fsc = RobustScaler()
    return fsc

def _min_max_scaler():
    fsc = MinMaxScaler((-1,1))
    return fsc

operator_map = {"fit_transform": _fit_transform,
                "transform": _transform,
                "StandardScaler": _standard_scaler(),
                "RobustScaler": _robust_scaler(),
                "MinMaxScaler":_min_max_scaler()}
# Map the function to the operators.
def select_scaler(**kwargs):

    # DEFINE SCALER
    fsc = StandardScaler()
    
    for key, value in kwargs.items(): 
        if key == 'scaler':
            try:
                fsc = operator_map[value]
            except:
                fsc = value
            break
    return fsc 

def mseed2norm(ths,**kwargs):
    fsc = select_scaler(**kwargs)    
    # RESHAPE TENSOR 3D-->2D
    nx, ny, nz = ths.shape
    ths = ths.reshape((nx,ny*nz)).T
    # APPLY SCALING METHOD
    for key, value in kwargs.items():
        if key == 'method':
            ths = operator_map[value](fsc,ths)
    
    # RESHAPE TENSOR 2D-->3D    
    ths = ths.T.reshape(nx,ny,nz)
    return ths,fsc

class thsTensorData(data_utils.Dataset):
    
    def __init__(self,inpX,inpY,inpZ,inpW,tar,idx,*inpR):
        self.inpX = inpX
        self.inpY = inpY
        self.inpZ = inpZ
        self.inpW = inpW
        self.tar  = tar
        self.idx  = idx
        self.inpR = inpR
    def __len__(self):
        return len(self.idx)
 
    def __getitem__(self,idx):        
        X = self.inpX[self.idx[idx],:,:]
        Y = self.inpY[self.idx[idx],:,:]
        Z = self.inpZ[self.idx[idx],:,:]
        W = self.inpW[self.idx[idx],:,:]
        y,z,w = self.tar
        y = y[self.idx[idx],:,:]
        z = z[self.idx[idx],:,:]
        w = w[self.idx[idx],:]
        if hasattr(self, 'inpR'):
            try:
                R0 = self.inpR[0][self.idx[idx],:,:]
                R1 = self.inpR[1][self.idx[idx],:,:]
                R2 = self.inpR[2][self.idx[idx],:,:]
                return X, Y, Z, W, y, z, w, R0, R1, R2
            except:
                return X, Y, Z, W, y, z, w
        else:
            return X, Y, Z, W, y, z, w

# Load mseed datasets downloaded via FDSN 
def load_mseed_dataset(source,inventory):
    stt = Stream()
    for src in source:
        stt += read(src)
    stt.sort(['channel'])
    inv = []
    for fld in inventory:
        if inv:
            inv += read_inventory(fld, "STATIONXML")
        else:
            inv = read_inventory(fld, "STATIONXML")
    return inv,stt

def corr_db(stt,inv,window):
    nsy = int(len(stt)/3.0)
    stt.detrend("demean")
    stt.detrend("linear")
    stt.taper(10.0/100.0)
    stt.remove_response(inventory=inv,output='ACC')
    
    stt.detrend("demean")
    stt.detrend("linear")
    stt.taper(10.0/100.0)
    
    nor_set = -999.9*np.ones(shape=(nsy,3,window))
    #
    trn_stt = stt.copy()
    tre = trn_stt.select(channel='H?E').sort(['network','station','starttime','location']) 
    trn = trn_stt.select(channel='H?N').sort(['network','station','starttime','location'])
    trz = trn_stt.select(channel='H?Z').sort(['network','station','starttime','location'])
    
    for i in range(0,nsy):
        tre[i].trim(tre[i].stats.starttime+30.,tre[i].stats.starttime+30.+(window-1)*tre[i].stats.delta)
        trn[i].trim(trn[i].stats.starttime+30.,trn[i].stats.starttime+30.+(window-1)*trn[i].stats.delta)
        trz[i].trim(trz[i].stats.starttime+30.,trz[i].stats.starttime+30.+(window-1)*trz[i].stats.delta)
    trn_stt.detrend("demean")
    trn_stt.detrend("linear")
    trn_stt.taper(10.0/100.0)
     
    for i in range(0,nsy):
        nor_set[i,0,:] = tre[i].data.reshape((window))
        nor_set[i,1,:] = trn[i].data.reshape((window))
        nor_set[i,2,:] = trz[i].data.reshape((window))
        
    vtm = np.arange(0,window)*1.0/stt[0].stats.sampling_rate 
    
    return stt,nor_set,vtm


def random_split(ths,lngs,idx=None):
    try: 
        len(idx)
    except:
        idx = rndp(sum(lngs))
    return idx,[Subset(ths,idx[off-lng:off]) 
        for off,lng in zip(_accumulate(lngs),lngs)]

def stead_dataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    
    vtm = md['dtm']*np.arange(0,md['ntm'])
    tar     = np.zeros((nsy,2))
    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pgat_set = -999.9*np.ones(shape=(nsy,3))
    psat_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    
    # parse hdf5 database
    eqd = h5py.File(src,'r')['earthquake']['local']
    eqm = pd.read_csv(osj(src.split('/waveforms')[0],'metadata_'+\
                          src.split('waveforms_')[-1].split('.hdf5')[0]+'.csv'))
    eqm = eqm.loc[eqm['trace_category'] == 'earthquake_local']
    eqm = eqm.loc[eqm['source_magnitude'] >= 3.5]
    eqm = eqm.sample(frac=nsy/len(eqm)).reset_index(drop=True)
    w = windows.tukey(md['ntm'],5/100)
    for i in eqm.index:
        tn = eqm.loc[i,'trace_name']
        bi = int(eqd[tn].attrs['p_arrival_sample'])
        for j in range(3):
            trn_set[i,j,:] = detrend(eqd[tn][bi:bi+Xwindow,j])*w
            pgat_set[i,j] = np.abs(trn_set[i,j,:]).max()
            trn_set[i,j,:] = trn_set[i,j,:]/pgat_set[i,j]
            pgat_set[i,j] = np.abs(trn_set[i,j,:]).max()
            _,psa,_,_,_ = rsp(md['dtm'],trn_set[i,j,:],md['vTn'],5.)
            psat_set[i,j,:] = psa.reshape((md['nTn']))
    pgat_set = np2t(np.float32(pgat_set))
    trn_set = np2t(trn_set).float()
    
    ths_fsc = []

    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    
    wnz_set   = tFT(nsy,nzd,zwindow)
    wnz_set.resize_(nsy,nzd,zwindow).normal_(**rndm_args)
    wnf_set   = tFT(nsy,nzf,zwindow)
    wnf_set.resize_(nsy,nzf,zwindow).normal_(**rndm_args)
    thf_set = lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])
    pgaf_set = -999.9*np.ones(shape=(nsy,3))
    psaf_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    
    for i in range(thf_set.shape[0]):
        for j in range(3):
            pgaf_set[i,j] = np.abs(thf_set[i,j,:].data.numpy()).max(axis=-1)
            _,psa,_,_,_ = rsp(md['dtm'],thf_set.data[i,j,:].numpy(),md['vTn'],5.)
            psaf_set[i,j,:] = psa.reshape((md['nTn']))
    pgat_set = np2t(np.float32(pgat_set))    
    pgaf_set = np2t(np.float32(pgaf_set))
    psat_set = np2t(np.float32(psat_set))
    psaf_set = np2t(np.float32(psaf_set))
        
    tar[:,0] = eqm['source_magnitude'].to_numpy(np.float32)
    tar[:,1] = eqm['source_depth_km'].to_numpy(np.float32)
    
    nhe = tar.shape[1] 
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar = np2t(np.float32(one_hot.fit_transform(tar)))
    from plot_tools import plot_ohe
    plot_ohe(tar)
    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}
    tar = (psat_set,psaf_set,tar)
    ths = thsTensorData(trn_set,thf_set,wnz_set,wnf_set,tar,partition['all'])

    # RANDOM SPLIT
    idx,\
    ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,\
    ths_tst,\
    ths_vld = ths_trn
    return ths_trn,ths_tst,ths_vld,vtm,fsc

def ann2bb_dataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    
    pbs = loadmat(src+"pbs.mat",squeeze_me=True,struct_as_record=False)['pbs']
    spm = loadmat(src+"spm.mat",squeeze_me=True,struct_as_record=False)['spm']
    rec = loadmat(src+"rec.mat",squeeze_me=True,struct_as_record=False)['rec']
    rec_pbs = loadmat(src+"rec_rec.mat",squeeze_me=True,struct_as_record=False)['rec']
    rec_spm = loadmat(src+"rec_spm.mat",squeeze_me=True,struct_as_record=False)['spm']

    nsy = min(nsy,pbs.mon.na,rec_pbs.mon.na)
    md['dtm'] =  round(rec_pbs.mon.dtm[0],3)
    md['dtm1'] = round(pbs.mon.dtm[0],3)
    vtm = md['dtm']*np.arange(0,md['ntm'])
    md['vtm1'] = md['dtm1']*np.arange(0,md['ntm'])
    w = windows.tukey(md['ntm'],5/100)
    
    tar     = np.zeros((nsy,2))
    rec_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pbs_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    spm_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    fil_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    sfm_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pga_rec_set = -999.9*np.ones(shape=(nsy,3))
    psa_rec_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    pga_fil_set = -999.9*np.ones(shape=(nsy,3))
    psa_fil_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    pga_pbs_set = -999.9*np.ones(shape=(nsy,3))
    psa_pbs_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    pga_spm_set = -999.9*np.ones(shape=(nsy,3))
    psa_spm_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    pga_sfm_set = -999.9*np.ones(shape=(nsy,3))
    psa_sfm_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    # parse mat database
    bi = 0
    pr = max(0,(Xwindow-len(rec.syn[0].tha.__dict__['ew'])))
    ps = max(0,(Xwindow-len(pbs.syn[0].tha.__dict__['ew'])))
    pr = divmod(pr,2)
    ps = divmod(ps,2)
    pr = (pr[0]+pr[1],pr[0])
    ps = (ps[0]+ps[1],ps[0])
    
    for i in range(nsy):
        for d,j in dirs.items():
            rec_tha                     = np.pad(rec.syn[i].tha.__dict__[d],pr)
            pbs_tha                     = np.pad(pbs.syn[i].tha.__dict__[d],ps)
            rec_pbs_tha             = np.pad(rec_pbs.syn[i].tha.__dict__[d],ps)
            spm_tha         = np.pad(spm.__dict__[d].syn[i].tha.__dict__[d],ps)    
            rec_spm_tha = np.pad(rec_spm.__dict__[d].syn[i].tha.__dict__[d],ps)
            #            
            rec_set[i,j,:] = detrend(rec_tha[bi:bi+Xwindow])*w
            pbs_set[i,j,:] = detrend(pbs_tha[bi:bi+Xwindow])*w
            spm_set[i,j,:] = detrend(spm_tha[bi:bi+Xwindow])*w
            fil_set[i,j,:] = detrend(rec_pbs_tha[bi:bi+Xwindow])*w
            sfm_set[i,j,:] = detrend(rec_spm_tha[bi:bi+Xwindow])*w
            #
#             from matplotlib import pyplot as plt
#             import seaborn as sns
#             clr = sns.color_palette("coolwarm",6)
#             _,hax=plt.subplots(nrows=3,ncols=2,sharex=True,sharey=True)
#             hax = list(hax)
#             hax[0][0].plot(pbs_set[i,j,:],label=r'$SPEED$',color=clr[0])
#             hax[1][0].plot(fil_set[i,j,:],label=r'$rec_{fil}^1$',color=clr[1])
#             hax[2][0].plot(filter_signal(rec_set[i,j,:].squeeze(),1.*2.*0.005),label=r'$rec_{fil}^2$',color=clr[2])
#             hax[0][1].plot(spm_set[i,j,:],label=r'$ANN2BB^1$',color=clr[3])
#             hax[1][1].plot(sfm_set[i,j,:],label=r'$ANN2BB^2$',color=clr[4])
#             hax[2][1].plot(rec_set[i,j,:],label='rec',color=clr[5])
#             hax[0][0].legend()
#             hax[1][0].legend()
#             hax[2][0].legend()
#             hax[0][1].legend()
#             hax[1][1].legend()
#             hax[2][1].legend()
#             plt.show()
            
            pga_rec_set[i,j] = np.abs(rec_set[i,j,:]).max()
            pga_pbs_set[i,j] = np.abs(pbs_set[i,j,:]).max()
            pga_fil_set[i,j] = np.abs(fil_set[i,j,:]).max()
            pga_spm_set[i,j] = np.abs(spm_set[i,j,:]).max()
            pga_sfm_set[i,j] = np.abs(sfm_set[i,j,:]).max()
#             _,psa,_,_,_ = rsp(md['dtm'],rec_set[i,j,:],md['vTn'],5.)
#             psar_set[i,j,:] = psa.reshape((md['nTn']))
#             _,psa,_,_,_ = rsp(md['dtm'],pbs_set[i,j,:],md['vTn'],5.)
#             psat_set[i,j,:] = psa.reshape((md['nTn']))
#             
    rec_set = np2t(rec_set).float()
    pbs_set = np2t(pbs_set).float()
    fil_set = np2t(fil_set).float()
    spm_set = np2t(spm_set).float()
    sfm_set = np2t(sfm_set).float()
    
    wnz_set   = tFT(nsy,nzd,zwindow)
    wnz_set.resize_(nsy,nzd,zwindow).normal_(**rndm_args)
    wnf_set   = tFT(nsy,nzf,zwindow)
    wnf_set.resize_(nsy,nzf,zwindow).normal_(**rndm_args)
    pbs_set = lowpass_biquad(pbs_set,1.5/0.01,md['cutoff'])
    
    for i in range(pbs_set.shape[0]):
        for j in range(3):
#             _,psa,_,_,_ = rsp(md['dtm'],pbs_set[i,j,:].data.numpy(),md['vTn'],5.)
#             psaf_set[i,j,:] = psa.reshape((md['nTn']))

            pga_pbs_set[i,j] = np.abs(pbs_set[i,j,:].data.numpy()).max()
            pga = np.max([pga_rec_set[i,j],pga_fil_set[i,j],\
                          pga_pbs_set[i,j],pga_spm_set[i,j],\
                          pga_sfm_set[i,j]])
            rec_set[i,j,:] = rec_set[i,j,:]/pga
            fil_set[i,j,:] = fil_set[i,j,:]/pga
            pbs_set[i,j,:] = pbs_set[i,j,:]/pga
            spm_set[i,j,:] = spm_set[i,j,:]/pga
            sfm_set[i,j,:] = sfm_set[i,j,:]/pga
            
    pga_rec_set = np2t(np.float32(pga_rec_set))    
    pga_fil_set = np2t(np.float32(pga_fil_set))
    psa_pbs_set = np2t(np.float32(psa_pbs_set))
    psa_spm_set = np2t(np.float32(psa_spm_set))
    psa_sfm_set = np2t(np.float32(psa_sfm_set))    
    ths_fsc = []

    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    tar[:,0] = np.float32(pbs.mon.dep[:nsy])
    tar[:,1] = np.float32(pbs.mon.dep[:nsy])
    
    nhe = tar.shape[1] 
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar = np2t(np.float32(one_hot.fit_transform(tar)))
    from plot_tools import plot_ohe
    plot_ohe(tar)
    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}
    tar = (psa_rec_set,psa_fil_set,tar)
    ths = thsTensorData(rec_set,fil_set,wnz_set,wnf_set,tar,partition['all'],pbs_set,spm_set,sfm_set)
    
    # RANDOM SPLIT
    idx,\
    ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,\
    ths_tst,\
    ths_vld = ths_trn
    return ths_trn,ths_tst,ths_vld,vtm,fsc,md
     
def synth_dataset(batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    
    # GENERATE SP SYNTHETICS
    
    ffr = md['cutoff']*2.*md['dtm']
    ths = sp(md,Xwindow)
    nor_set = ths.generate(nsy)
    nor_set+= ths.generate(nsy)
    nor_set+= ths.generate(nsy)
    pad = 0#int(1.5*2*2/ffr)
    md['ntm']=Xwindow
    if pad>0:
        md['ntm'] += nxtpow2(Xwindow+2*pad)
        pad = int((md['ntm']-Xwindow)/2)
        for i in range(0,3*nsy):
            nor_set[i] = np.pad(nor_set[i],pad,'constant')
    
    vtm = md['dtm']*np.arange(0,md['ntm'])
    # RESTRUCTURE VECTORS TO TENSORS
    tar     = np.zeros((nsy,2))
    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pgat_set = -999.9*np.ones(shape=(nsy,3))
    psat_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))

    for i in range(0,nsy):
        trn_set[i,0,:] = nor_set[3*i+0][:md['ntm']]
        trn_set[i,1,:] = nor_set[3*i+1][:md['ntm']]
        trn_set[i,2,:] = nor_set[3*i+2][:md['ntm']]
        pgat_set[i,0] = np.abs(trn_set[i,0,:]).max()
        pgat_set[i,1] = np.abs(trn_set[i,1,:]).max()
        pgat_set[i,2] = np.abs(trn_set[i,2,:]).max()
        _,psa0,_,_,_ = rsp(md['dtm'],trn_set[i,0,:],md['vTn'],5.)
        _,psa1,_,_,_ = rsp(md['dtm'],trn_set[i,1,:],md['vTn'],5.)
        _,psa2,_,_,_ = rsp(md['dtm'],trn_set[i,2,:],md['vTn'],5.)
        #psa0=psa0/psa0[0]
        #psa0=psa0/psa0[1]
        #psa0=psa0/psa0[2]
        psat_set[i,0,:] = psa0.reshape((md['nTn']))
        psat_set[i,1,:] = psa1.reshape((md['nTn']))
        psat_set[i,2,:] = psa2.reshape((md['nTn']))
       
    pgat_set = np2t(np.float32(pgat_set))
    trn_set = np2t(trn_set).float()
    # COMPUTE ARIAS INTENSITY
    #_,_,ait05 = arias_intensity(md['dtm'],trn_set.reshape((-1,md['ntm'])),pc=0.05)
    #_,_,ait95 = arias_intensity(md['dtm'],trn_set.reshape((-1,md['ntm'])),pc=0.95)
    #md['DS595'] = ait95.reshape(*trn_set.shape[:-1])-ait05.reshape(*trn_set.shape[:-1])
    #DS595 = ait95.reshape(*trn_set.shape[:-1])-ait05.reshape(*trn_set.shape[:-1])
    ths_fsc = []

    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    
    wnz_set   = tFT(nsy,nzd,zwindow)
    wnz_set.resize_(nsy,nzd,zwindow).normal_(**rndm_args)
    wnf_set   = tFT(nsy,nzf,zwindow)
    wnf_set.resize_(nsy,nzf,zwindow).normal_(**rndm_args)
    thf_set  = flt(dtm=md['dtm'],ths=trn_set,ffr=ffr)
    pgaf_set = -999.9*np.ones(shape=(nsy,3))
    psaf_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    
    for i in range(thf_set.shape[0]):
        trn_set[i,0,:] = trn_set[i,0,:]/pgat_set[i,0]
        trn_set[i,1,:] = trn_set[i,1,:]/pgat_set[i,1]
        trn_set[i,2,:] = trn_set[i,2,:]/pgat_set[i,2]
        pgaf_set[i,0] = np.abs(thf_set[i,0,:].data.numpy()).max(axis=-1)
        pgaf_set[i,1] = np.abs(thf_set[i,1,:].data.numpy()).max(axis=-1)
        pgaf_set[i,2] = np.abs(thf_set[i,2,:].data.numpy()).max(axis=-1)
        _,psa0,_,_,_ = rsp(md['dtm'],thf_set.data[i,0,:].numpy(),md['vTn'],5.)
        _,psa1,_,_,_ = rsp(md['dtm'],thf_set.data[i,1,:].numpy(),md['vTn'],5.)
        _,psa2,_,_,_ = rsp(md['dtm'],thf_set.data[i,2,:].numpy(),md['vTn'],5.)
        #psa0=psa0/psa0[0]
        #psa0=psa0/psa0[1]
        #psa0=psa0/psa0[2]
        psaf_set[i,0,:] = psa0.reshape((md['nTn']))
        psaf_set[i,1,:] = psa1.reshape((md['nTn']))
        psaf_set[i,2,:] = psa2.reshape((md['nTn']))
        thf_set[i,0,:] = thf_set[i,0,:]/pgat_set[i,0]
        thf_set[i,1,:] = thf_set[i,1,:]/pgat_set[i,1]
        thf_set[i,2,:] = thf_set[i,2,:]/pgat_set[i,2]
    
    pgat_set = np2t(np.float32(pgat_set))    
    pgaf_set = np2t(np.float32(pgaf_set))
    psat_set = np2t(np.float32(psat_set))
    psaf_set = np2t(np.float32(psaf_set))
    
    tar[:,0] = md['mw']
    tar[:,1] = md['dep']
    
    nhe = tar.shape[1] 
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar = np2t(np.float32(one_hot.fit_transform(tar)))
    from plot_tools import plot_ohe
    plot_ohe(tar)
    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}
    tar = (psat_set,psaf_set,tar)
    ths = thsTensorData(trn_set,thf_set,wnz_set,wnf_set,tar,partition['all'])

    # RANDOM SPLIT
    idx,\
    ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,\
    ths_tst,\
    ths_vld = ths_trn
    return ths_trn,ths_tst,ths_vld,vtm,fsc

def load_dataset(batch_percent,source=['./signals/*.*.*.mseed'],\
                 inventory=['./signals/sxml/RM07.xml',
                            './signals/sxml/LXRA.xml',
                            './signals/sxml/SRN.xml']):
    u'''LOAD DATASET'''
    inv,stt = load_mseed_dataset(source,inventory)    
    window = 1024 #1087 #3500
    stt,trn_set,vtm = corr_db(stt,inv,window)
    nsy = int(len(stt)/3.0)
    dtm=vtm[1]-vtm[0]
    md['ntm']=window
    
#     nor_set,_ = mseed2norm(nor_set[:,:,:],\
#                            method='fit_transform',\
#                            scaler='StandardScaler')
    #tst_set , = mseed2norm(trn_set[:limit,:,:],method='fit_transform')
        
    pgat_set = -999.9*np.ones(shape=(nsy,3,))
    for i in range(0,nsy):
        pgat_set[i,0] = np.abs(trn_set[i,0,:]).max()
        pgat_set[i,1] = np.abs(trn_set[i,1,:]).max()
        pgat_set[i,2] = np.abs(trn_set[i,2,:]).max()
        trn_set[i,0,:] = trn_set[i,0,:]/pgat_set[i,0]
        trn_set[i,1,:] = trn_set[i,1,:]/pgat_set[i,1]
        trn_set[i,2,:] = trn_set[i,2,:]/pgat_set[i,2]
    # TRAINING SET
    trn_set = np2t(trn_set)
    trn_set = trn_set.float()

    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    
    wnz = gn(pc=2e-1,ntm=window,scl=1.)
    wnz = flg(wnz=wnz,ffr=5.*2.*dtm)
    
    thf_set = flt(dtm=dtm,ths=trn_set,\
                  ffr=5.*2.*dtm,\
                  pc=2e-1,wnz=wnz)
    pgaf_set = -999.9*np.ones(shape=(nsy,3))
    for i in range(thf_set.shape[0]):
        pgaf_set[i,0] = np.abs(thf_set[i,0,:].data.numpy()).max(axis=-1)
        pgaf_set[i,1] = np.abs(thf_set[i,1,:].data.numpy()).max(axis=-1)
        pgaf_set[i,2] = np.abs(thf_set[i,2,:].data.numpy()).max(axis=-1)
        thf_set[i,0,:] = thf_set[i,0,:]/pgaf_set[i,0]
        thf_set[i,1,:] = thf_set[i,1,:]/pgaf_set[i,1]
        thf_set[i,2,:] = thf_set[i,2,:]/pgaf_set[i,2]
    tar = np.ones((nsy,2),dtype=np.float32)
    ths_fsc = []
    #tar[:,0] = np.float32(md['mw'])
    #tar[:,1] = np.float32(md['dep'])
    
    # ONE HOT VECTOR
    nhe = tar.shape[1] 
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    one_hot = OneHotEncoder()
    tar = one_hot.fit_transform(tar).toarray()
    tar = np.float32(tar)#np.int64(tar)
#     from plot_tools import plot_ohe
#     plot_ohe(tar)

    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}
    ths = thsTensorData(trn_set,(pgat_set,tar),partition['all'])
    thf = thsTensorData(thf_set,(pgaf_set,tar),partition['all'])
    # RANDOM SPLIT
    idx,\
    ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,\
    ths_tst,\
    ths_vld = ths_trn
    
    _,\
    thf_trn = random_split(thf,[trn,vld,tst],idx)
    thf_trn,\
    thf_tst,\
    thf_vld = thf_trn    
    
    return ths_trn,ths_tst,ths_vld,\
           thf_trn,thf_tst,thf_vld,\
           vtm,fsc

def arias_intensity(dtm,tha,pc=0.95,nf=9.81):
    aid = np.pi/2./nf*cumtrapz(tha**2, dx=dtm, axis=-1, initial = 0.)
    mai = np.max(aid,axis=-1)
    ait = np.empty_like(mai)
    idx = np.empty_like(mai)
    if mai.size>1:
        for i in range(mai.size):
            ths = np.where(aid[i,...]/mai[i]>=pc)[0][0]
            ait[i] = aid[i,ths]
            idx[i] = ths*dtm
    else:
        ths = np.where(aid/mai>=pc)[0][0]
        ait = aid[ths]
        idx = ths*dtm
    return aid,ait,idx

def deepbns_dataset(src,batch_percent,xwindow,zwindow,nzd,nzf,md,nsy,device):
    grp = list(h5py.File(src,'r+').keys())
    nsy = min(nsy,80)
    trn_set = np.zeros((nsy,3,4096))   ## Should we resize the dimensions to 1D ?and  the length is not 4096
    pga_set = np.empty((nsy,3,1))
    tar     = np.zeros((nsy,2))

    for i,g in zip(range(nsy),grp):
        trn_set[i,0,:] = np.nan_to_num(pd.read_hdf(src,"{}/hc_timeseries".format(g)))
        trn_set[i,1,:] = np.nan_to_num(pd.read_hdf(src,"{}/hc_timeseries".format(g)))
        trn_set[i,2,:] = np.nan_to_num(pd.read_hdf(src,"{}/hc_timeseries".format(g)))
        tar[i,0] = pd.read_hdf(src,"{}/metadata".format(g))['Mchirp'] 
        tar[i,1] = pd.read_hdf(src,"{}/metadata".format(g))['Ltilde']

    nhe = tar.shape[1] 
    ths_fsc=[]
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    one_hot = OneHotEncoder()
    tar = one_hot.fit_transform(tar).toarray()
    tar = np2t(np.float32(tar)) #np.int64(tar)
    pga_set = np2t(np.float32(pga_set)) #np.int64(tar)

    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}


    
    trn_set = np2t(trn_set)
    trn_set = trn_set.float()
    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    
    rndm_args = {'mean': 0, 'std': 1}

    wnz_set   = tFT(nsy,nzd,zwindow)
    wnz_set.resize_(nsy,nzd,zwindow).normal_(**rndm_args)
    wnf_set   = tFT(nsy,nzf,zwindow)
    wnf_set.resize_(nsy,nzf,zwindow).normal_(**rndm_args)
    ffr = md['cutoff']*2.*md['dtm']
    thf_set  = flt(dtm=md['dtm'],ths=trn_set,ffr=ffr)

    for i in range(thf_set.shape[0]):
        pga = np.abs(trn_set[i,0,:].data.numpy()).max(axis=-1)
        trn_set[i,:,:] = thf_set[i,:,:]/pga
        thf_set[i,:,:] = thf_set[i,:,:]/pga
        pga_set[i,:,0].fill_(pga)

    tar = (pga_set,pga_set,tar)
    ths = thsTensorData(trn_set,thf_set,wnz_set,wnf_set,tar,partition['all'])

    # RANDOM SPLIT
    idx,\
    ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,\
    ths_tst,\
    ths_vld = ths_trn
    vtm = md['dtm']*np.arange(0,md['ntm'],1)
    return ths_trn,ths_tst,ths_vld,vtm,fsc

# [TODO]
def civa_dataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    '''
    Input:
        src: source file
        batch_percent: % for trn/tst/vld
        Xwindow: number of time steps (including 0) in the time-window (default 4096)
        zwindow: length of the latent space last dimension
        nzd: number of channels of broad-band latent space (Xd<->zd)
        nzf: number of channels of filtered latent space (Xf<->zf)
        md: meta-data dictionary
        nsy: number of signals to extract from the src database
        device: CPU or GPU

    Output: 
        ths_trn: training set dataloader 
        ths_tst: test set dataloader
        ths_vld: validation set dataloader
        vtm: time-vector (vtm = np.arange(0,Xwindow,md['dtm'],dtype=np.float32)

    Observations: 
        1. Latent space shape:
            zd.shape = (nsy,nzd,zwindow) and zf.shape = (nsy,nzf,zwindow)
        2. Each output dataloader (ths_trn,ths_tst,ths_vld) can be unwrapped as:

            for b,batch in enumerate(ths_trn):
                # Load batch (enumerating the dataloader ths_trn)
                xd_data,xf_data,zd_data,zf_data,_,_,_ = batch
                Xd = Variable(xd_data).to(device) # broad-band signal converted to pytorch Variable
                zd = Variable(zd_data).to(device) # broad-band latent space converted to pytorch Variable)
                Xf = Variable(xf_data).to(device) # filtered signal converted to pytorch Variable
                zf = Variable(zf_data).to(device) # filtered latent space converted to pytorch Variable)
                ...
    '''
    # time vector
    vtm = md['dtm']*np.arange(0,md['ntm'])
    # initialize numpy tensors 
    tar = np.zeros((nsy,2))
    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pgat_set = -999.9*np.ones(shape=(nsy,3))
    psat_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    
    # parse CIVA database
    eqd = [...]  # h5py.File(src,'r')['earthquake']['local']
    # parse CIVA metadata
    eqm = [...]
            #pd.read_csv(osj(src.split('/waveforms')[0],'metadata_'+\
            #                src.split('waveforms_')[-1].split('.hdf5')[0]+'.csv'))

    # Tukey window
    w = windows.tukey(md['ntm'],5/100)
    for i in nsy:
        # [TODO]
        # read 3-component time-histories into a numpy tensor civa_ths
        civa_ths = np.empty((nsy,3)) # must be populated with dataset from civa
        [...]
        # loop over components
        bi = 0 # initial index of the time history (can be changed)
        for j in range(3):
            # detrend portion of time
            trn_set[i,j,:] = detrend(civa_ths[bi:bi+Xwindow,j])*w
            # compute peak values per component
            pgat_set[i,j] = np.abs(trn_set[i,j,:]).max()
            # normalize
            trn_set[i,j,:] = trn_set[i,j,:]/pgat_set[i,j]
            # recompute peak value per component
            pgat_set[i,j] = np.abs(trn_set[i,j,:]).max()
            # response spectrum ---> not necessary
            #_,psa,_,_,_ = rsp(md['dtm'],trn_set[i,j,:],md['vTn'],5.)
            #psat_set[i,j,:] = psa.reshape((md['nTn']))
    
    # convert numpy tensors to pytorch tensors
    pgat_set = np2t(np.float32(pgat_set))
    trn_set = np2t(trn_set).float()
    
    # Define database partitioning
    ths_fsc = []
    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    
    # define latent space (random values from normal centered distribution)
    wnz_set   = tFT(nsy,nzd,zwindow)
    wnz_set.resize_(nsy,nzd,zwindow).normal_(**rndm_args)
    wnf_set   = tFT(nsy,nzf,zwindow)
    wnf_set.resize_(nsy,nzf,zwindow).normal_(**rndm_args)

    # filter original CIVA time-histories (md['cutoff'] is the cutoff frequency)
    thf_set = lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])

    # Normalize filtered data
    pgaf_set = -999.9*np.ones(shape=(nsy,3))
    psaf_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    for i in range(thf_set.shape[0]):
        for j in range(3):
            pgaf_set[i,j] = np.abs(thf_set[i,j,:].data.numpy()).max(axis=-1)
            #_,psa,_,_,_ = rsp(md['dtm'],thf_set.data[i,j,:].numpy(),md['vTn'],5.)
            #psaf_set[i,j,:] = psa.reshape((md['nTn']))
    # Convert numpy tensor to pytorch tensors
    pgat_set = np2t(np.float32(pgat_set))    
    pgaf_set = np2t(np.float32(pgaf_set))
    psat_set = np2t(np.float32(psat_set))
    psaf_set = np2t(np.float32(psaf_set))
       
    # Define target (fake labels issued from metadata)
    tar[:,0] = np.empty(dtype=np.float32) # source magnitude
    tar[:,1] = np.empty(dtype=np.float32) # source depth
    
    nhe = tar.shape[1] 
    # Scaler for metadata homogeneization
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar = np2t(np.float32(one_hot.fit_transform(tar)))
    from plot_tools import plot_ohe
    plot_ohe(tar)
    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}
    tar = (psat_set,psaf_set,tar)
    ths = thsTensorData(trn_set,thf_set,wnz_set,wnf_set,tar,partition['all'])

    # RANDOM SPLIT
    idx,\
    ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,\
    ths_tst,\
    ths_vld = ths_trn
    return ths_trn,ths_tst,ths_vld,vtm,fsc
