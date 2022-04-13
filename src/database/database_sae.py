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
import dask
import dask.dataframe as dd
import dask.array as da
from dask import delayed
import random as rnd
from os.path import join as opj
from torch import from_numpy as np2t
from common.common_nn import to_categorical
from common.common_torch import o1l
from torch.utils import data as data_utils
from torch import randperm as rndp
from torch.utils.data.dataset import Subset, _accumulate
from torch.utils.data import DataLoader as dloader
from obspy import Stream, read, read_inventory
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tools.generate_noise as gng
from tools.generate_noise import filter_ths_tensor as flt, filter_signal
from tools.generate_noise import lowpass_biquad
from tools.generate_noise import filter_Gauss_noise as flg
from tools.generate_noise import Gauss_noise as gn
from tools.generate_noise import pollute_ths_tensor as pgn
from database.synthetic_generator import sp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.integrate import cumtrapz
from torch import FloatTensor as tFT
from database.ss_process import rsp
from scipy.signal import detrend, windows
from scipy.io import loadmat

#from dask.distributed import Client
#client = Client(n_workers=1, threads_per_worker=4, processes=False, memory_limit='2GB')

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

class H5ls:
    def __init__(self,nsy=0,names=[],xw=1):
        # Store an empty list for dataset names
    
        self.names = names
        self.nsy= nsy-1
        self.xw = xw
        self.w = windows.tukey(xw,5./100.).astype(np.float32)
        self.ths = np.zeros(shape=(nsy,3,xw),dtype=np.float32)
        self.pga = np.zeros(shape=(nsy,3),dtype=np.float32)
 
    def __call__(self, name, h5obj):
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(h5obj,'dtype') and \
            not name in self.names and \
            self.nsy>=0:

            self.names.append(name)
            bi = int(h5obj.attrs['p_arrival_sample'])
            for j in range(3):
                ths = h5obj[bi:bi+self.xw,j].astype(np.float32)
                self.ths[self.nsy,j,:len(ths)] = detrend(ths)*self.w[:len(ths)]
                self.pga[self.nsy,j]   = np.abs(self.ths[self.nsy,j,:]).max().astype(np.float32)
                self.ths[self.nsy,j,:] /= self.pga[self.nsy,j]
                self.nsy-=1

def STEADdatasetMPI(comm,size,rank,src,batch_percent,workers,
    Xwindow,zwindow,nzd,nzf,md,nsy,gather=False):
    meta = {'network_code':'str','receiver_code':'str','receiver_type':'str',
        'receiver_latitude':np.float64,'receiver_longitude':np.float64,'receiver_elevation_m':np.float64,
        'p_arrival_sample':np.float64,'p_status':'str','p_weight':np.float64,'p_travel_sec':np.float64,
        's_arrival_sample':np.float64,'s_status':'str','s_weight':np.float64,
        'source_id':'str','source_origin_time':'str',
        'source_origin_uncertainty_sec':np.float64,'source_latitude':np.float64,'source_longitude':np.float64,
        'source_error_sec':np.float64,'source_gap_deg':np.float64,
        'source_horizontal_uncertainty_km':np.float64,'source_depth_km':np.float64,
        'source_depth_uncertainty_km':np.float64,'source_magnitude':np.float64,
        'source_magnitude_type':'str','source_magnitude_author':'str',
        'source_mechanism_strike_dip_rake':'str','source_distance_deg':np.float64,
        'source_distance_km':np.float64,'back_azimuth_deg':np.float64,'snr_db':'str',
        'coda_end_sample':np.float64,'trace_start_time':'str','trace_category':'str',
        'trace_name':'str'}

    print("STEADdatasetMPI ...")

    vtm = md['dtm']*np.arange(0,md['ntm'])
    
    tar     = np.zeros((nsy,2))

    eqm = dd.read_csv(urlpath=src.replace('.hdf5','.csv').replace('waveforms','metadata'),
        na_values='None',dtype=meta).query('trace_category == "earthquake_local"').query('source_magnitude>=3.5')
    
    
    eqm = eqm.sample(frac=nsy/len(eqm),replace=True).reset_index(drop=True).set_index("trace_name",sorted=True)
    
    nsy_set = len(eqm)//size

    nsy = nsy_set*size
    print("priors calculations ...")
    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']),dtype=np.float32)
    thf_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']),dtype=np.float32)

    pgat_set = -999.9*np.ones(shape=(nsy,3),dtype=np.float32)
    pgaf_set = -999.9*np.ones(shape=(nsy,3),dtype=np.float32)

    psat_set = -999.9*np.ones(shape=(1,1,1),dtype=np.float32)
    psaf_set = -999.9*np.ones(shape=(1,1,1),dtype=np.float32)
    
    trn_set  = -999.9*np.ones(shape=(nsy_set,3,md['ntm']),dtype=np.float32)
    thf_set  = -999.9*np.ones(shape=(nsy_set,3,md['ntm']),dtype=np.float32)

    pgat_set = -999.9*np.ones(shape=(nsy_set,3),dtype=np.float32)
    pgaf_set = -999.9*np.ones(shape=(nsy_set,3),dtype=np.float32)
    
    h5ls = H5ls(nsy=nsy_set,
        names=eqm.index.compute().to_list()[rank*nsy_set:(rank+1)*nsy_set],
        xw=Xwindow)

    print("preparing to extracting files {}".format(src))

    f = h5py.File(src,'r',driver='mpio',comm=comm)
    print("parallel extraction ...\n")
    dsets = f['earthquake']['local']
    # comm.barrier()
    # this will now visit all objects inside the hdf5 file and store datasets in h5ls.names
    dsets.visititems(h5ls)
    trn_set  = h5ls.ths_trn
    pgat_set = h5ls.pga
    print("extracting hdf5 files ...")

        

    trn_set  = np2t(np.float32(trn_set))
    pgat_set = np2t(np.float32(pgat_set))

    thf_set = lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])

    for i in range(thf_set.shape[0]):
        for j in range(3):
            pgaf_set[i,j] = np.abs(thf_set[i,j,:].data.numpy()).max(axis=-1)
            #_,psa,_,_,_ = rsp(md['dtm'],thf_set.data[i,j,:].numpy(),md['vTn'],5.)
            # psaf_set[i,j,:] = psa.reshape((md['nTn']))
            
    pgat_set = np2t(np.float32(pgat_set))    
    pgaf_set = np2t(np.float32(pgaf_set))
    # psat_set = np2t(np.float32(psat_set))
    # psaf_set = np2t(np.float32(psaf_set))

    wnz_set = tFT(nsy_set,nzd,zwindow).resize_(nsy_set,nzd,zwindow).normal_(**rndm_args)
    wnf_set = tFT(nsy_set,nzf,zwindow).resize_(nsy_set,nzf,zwindow).normal_(**rndm_args)
        
    tar = eqm[['source_magnitude','source_depth_km']].compute().to_numpy(np.float32)
    nhe = tar.shape[1] 
    tar_fsc = select_scaler(method='fit_transform',\
                            scaler='StandardScaler')
    tar_set = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar_set[:,i] = lab_enc.fit_transform(tar_set[:,i])
    from sklearn.preprocessing import MultiLabelBinarizer
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar_set = np2t(np.float32(one_hot.fit_transform(tar_set)))
    # from plot_tools import plot_ohe
    # plot_ohe(tar)

    ths_fsc = []
    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,
           'ths_fsc':ths_fsc,
           'one_hot':one_hot,
           'ncat':tar.shape[1]}

    tar = (psat_set,psaf_set,tar)

    if gather:
        md['nsy'] = nsy
        trn_set =  comm.gather(trn_set,root=0)
        thf_set =  comm.gather(thf_set,root=0)
        wnz_set =  comm.gather(wnz_set,root=0)
        wnf_set =  comm.gather(wnf_set,root=0)
        pgat_set = comm.gather(pgat_set,root=0)
        pgaf_set = comm.gather(pgaf_set,root=0)
        tar =  comm.gather(tar_set,root=0)

        if rank == 0:
            trn_set = torch.vstack(trn_set)
            thf_set = torch.vstack(thf_set)
            wnz_set = torch.vstack(wnz_set)
            wnf_set = torch.vstack(wnf_set)
            pgat_set = torch.vstack(pgat_set)
            pgaf_set = torch.vstack(pgaf_set)
            tar = torch.vstack(tar)
    
            partition = {'all': range(0,nsy)}
            trn = max(1,int(batch_percent[0]*nsy))
            tst = max(1,int(batch_percent[1]*nsy))
            vld = max(1,nsy-trn-tst)

            ths = thsTensorData(trn_set,thf_set,wnz_set,wnf_set,tar,partition['all'])

            # RANDOM SPLIT
            idx,\
            ths_trn = random_split(ths,[trn,vld,tst])
            ths_trn,\
            ths_tst,\
            ths_vld = ths_trn
               
            return ths_trn,ths_tst,ths_vld,vtm,fsc
        else:
            return None
    else:
        md['nsy'] = nsy_set
        partition = {'all': range(0,nsy)}
        trn = max(1,int(batch_percent[0]*nsy))
        tst = max(1,int(batch_percent[1]*nsy))
        vld = max(1,nsy-trn-tst)

        ths = thsTensorData(trn_set,thf_set,wnz_set,wnf_set,tar,partition['all'])

        # RANDOM SPLIT
        idx,\
        ths_trn = random_split(ths,[trn,vld,tst])
        ths_trn,\
        ths_tst,\
        ths_vld = ths_trn
           
        return ths_trn,ths_tst,ths_vld,vtm,fsc
    
    # index_counts = eqm.map_partitions(lambda _df: _df.index.value_counts().sort_index()).compute()
    # index = np.repeat(index_counts.index, index_counts.values)
    # divisions, _ = dd.io.io.sorted_division_locations(index, npartitions=eqm.npartitions)
    # eqm = eqm.repartition(divisions=divisions).persist()

        
            #_,psa,_,_,_ = rsp(md['dtm'],trn_set[i,j,:],md['vTn'],5.)
            #psat_set[i,j,:] = psa.reshape((md['nTn']))        
    #edq = dd.read_hdf(pattern=src,key='/earthquake/local',mode='r+')

    # # Find out where data is on each worker
    # key_to_part_dict = {str(part.key): part for part in futures_of(eqm)}
    # who_has = client.who_has(eqm)
    # worker_map = defaultdict(list)
    # for key, workers in who_has.items():
    #     worker_map[first(workers)].append(key_to_part_dict[key])
    # # Call an MPI-enabled function on the list of data present on each worker
    # futures = [client.submit(print_data_and_rank,
    #     list_of_parts,workers=worker) 
    #     for worker, list_of_parts in worker_map.items()]

    # wait(futures)

    # client.close()
    

def STEADdataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    print('Enter in the stead_dataset function ...') 
    vtm = md['dtm']*np.arange(0,md['ntm'])
    tar     = np.zeros((nsy,2))
    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pgat_set = -999.9*np.ones(shape=(nsy,3))
    psat_set = -999.9*np.ones(shape=(1,1,1))
    print("training data ",trn_set.shape,pgat_set.shape, psat_set.shape)
    print("src ", src)
    # parse hdf5 database
    eqd = h5py.File(src,'r')['earthquake']['local']
    eqm = pd.read_csv(opj(src.split('/waveforms')[0],'metadata_'+\
                          src.split('waveforms_')[-1].split('.hdf5')[0]+'.csv'))
    eqm = eqm.loc[eqm['trace_category'] == 'earthquake_setal']
    eqm = eqm.loc[eqm['source_magnitude'] >= 3.5]
    eqm = eqm.sample(frac=nsy/len(eqm)).reset_index(drop=True)
    w = windows.tukey(md['ntm'],5/100)

    for i in eqm.index:
        tn = eqm.loc[i,'trace_name']
        bi = int(eqd[tn].attrs['p_arrival_sample'])
        for j in range(3):
            print("in the loop i,j",i,j)
            print("dimensions of eqd passed in the program",tn,bi,j,Xwindow)
            a = eqd[tn][bi:bi+Xwindow,j]
            #a = np.reshape(a.shape[0],1)
            #replace nan value to zeros if datatset is corrupted
            print("shape of array", a.shape)
            print("number of nan in the array to detrend", np.isnan(a).sum())
            b = detrend(a)
            trn_set[i,j,:] = b*w
            print("__after__ detrend function")
            pgat_set[i,j] = np.abs(trn_set[i,j,:]).max()
            trn_set[i,j,:] = trn_set[i,j,:]/pgat_set[i,j]
            pgat_set[i,j] = np.abs(trn_set[i,j,:]).max()
            #_,psa,_,_,_ = rsp(md['dtm'],trn_set[i,j,:],md['vTn'],5.)
            #psat_set[i,j,:] = psa.reshape((md['nTn']))
    pgat_set = np2t(np.float32(pgat_set))
    trn_set = np2t(trn_set).float()
    
    ths_fsc = []

    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)
    
    wnz_set   = tFT(nsy,nzd,zwindow)
    wnz_set.resize_(nsy,nzd,zwindow).normal_(**rndm_args)
    #extracting a submatrix of wnz_set bot style a Normal distribution
    wnf_set   = wnz_set[:,::nzd//nzf,:]
    # wnf_set   = tFT(nsy,nzf,zwindow)
    # wnf_set.resize_(nsy,nzf,zwindow).normal_(**rndm_args)

    #[TODO] change thf_set 
    # thf_set = trn_set #lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])
    
    thf_set  = lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])
    pgaf_set = -999.9*np.ones(shape=(nsy,3))
    psaf_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    
    for i in range(thf_set.shape[0]):
        for j in range(3):
            pgaf_set[i,j] = np.abs(thf_set[i,j,:].data.numpy()).max(axis=-1)
            _,psa,_,_,_ = rsp(md['dtm'],thf_set.data[i,j,:].numpy(),md['vTn'],5.)
            psaf_set[i,j,:] = psa.reshape((md['nTn']))
            print("psa proceed ...",i,j)
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
    print("values ....")
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
            #pd.read_csv(opj(src.split('/waveforms')[0],'metadata_'+\
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


def mdof_dataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,ntm,mdof,wdof,tdof,wtdof,md,nsy,device):
    '''
    Input:
        src: source file
        batch_percent: % for trn/tst/vld
        Xwindow: number of time steps (including 0) in the time-window (default 4096)
        zwindow: length of the latent space last dimension
        nzd: number of channels of broad-band latent space (Xd<->zd)
        nzf: number of channels of filtered latent space (Xf<->zf)
        ntm: number of time series
        mdof: number of channels
        wdof: channel numbers (int)
        tdof: type of channels (e.g. A)
        wtdof: specify the subdivision of the channels in tdofs
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

    # Initialize wtdof_v tensor
    for i1 in range(len(wtdof)):
        if i1==0:
            wtdof_v=[wtdof[i1]]
            wtdof_v=np.array(wtdof_v)
            np.expand_dims(wtdof_v, axis=0)
        else:
            i2=wtdof_v[i1-1]+wtdof[i1]
            np.concatenate((wtdof_v,i2), axis=1)

    # time vector
    vtm = md['dtm']*np.arange(0,md['ntm'])
    # initialize numpy tensors 
    tar = np.zeros((nsy,1))
    trn_set  = -999.9*np.ones(shape=(nsy,mdof,md['ntm']))
    thf_set  = -999.9*np.ones(shape=(nsy,mdof,md['ntm']))
    # initialize numpy tensor for normalizing original (undamaged) data
    pgat_set = -999.9*np.ones(shape=(nsy,mdof))
    psat_set = -999.9*np.ones(shape=(nsy,mdof,md['nTn']))
    # initialize numpy tensor for normalizing filtered (damaged) data
    pgaf_set = -999.9*np.ones(shape=(nsy,3))
    psaf_set = -999.9*np.ones(shape=(nsy,3,md['nTn']))
    
    case_ud='damaged'
    [mdof_ths_u,eqm_u]=mdof_case_loader(mdof,wtdof_v,src,case_ud,tdof,wdof,nsy,ntm)
    [trn_set,pgat_set]=mdof_tuk_pyt(md,nsy,mdof,mdof_ths_u,Xwindow,trn_set,pgat_set)

    case_ud='undamaged'
    [mdof_ths_d,eqm_d]=mdof_case_loader(mdof,wtdof_v,src,case_ud,tdof,wdof,nsy,ntm)
    [thf_set,pgaf_set]=mdof_tuk_pyt(md,nsy,mdof,mdof_ths_d,Xwindow,thf_set,pgaf_set)

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

    # Define target (fake labels issued from metadata)  **damaged only**
    tar[:,0]=eqm_d
    
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

# for mdof only ########################################################################
def mdof_case_loader(mdof,wtdof_v,src,case_ud,tdof,wdof,nsy,ntm):
    # parse MDOF database
    i4=0
    for i1 in range(mdof):
        if i1 > wtdof_v[i4]:
            i4=i4+1
        #load the measurements recorded by each single channel
        if len(wtdof_v)>1:
            src_dof=opj(src,"{:>s}_{:>s}_DC_concat_dof_{:>d}.csv".format(case_ud,tdof[i4],wdof[i1]))
        else:
            src_dof=opj(src,"{:>s}_{:>s}_DC_concat_dof_{:>d}.csv".format(case_ud,tdof,wdof[i1]))
        sdof=np.genfromtxt(src_dof)
        sdof.astype(np.float32)

        #initialise mdof_ths
        if i1==0:
            mdof_ths=np.zeros((nsy,mdof,ntm))
        i2=0
        for i3 in range(nsy):
            mdof_ths[i3,i1,0:ntm]=sdof[i2:(i2+ntm)]
            i2=i2+ntm

    # parse MDOF metadata

    src_metadata=opj(src,"{:>s}_DC_labels.csv".format(case_ud))
    eqm=np.genfromtxt(src_metadata)
    return mdof_ths,eqm
# for mdof only ########################################################################

# for mdof only ########################################################################
def mdof_tuk_pyt(md,nsy,mdof,mdof_ths,Xwindow,trn_set,pgat_set):
    # Tukey window
    w = windows.tukey(md['ntm'],5/100)
    for i in range(nsy):
        # loop over components
        bi = 0 # initial index of the time history (can be changed)
        for j in range(mdof):
            # detrend portion of time
            trn_set[i,j,:] = detrend(mdof_ths[i,j,bi:bi+Xwindow])*w 
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
    return trn_set,pgat_set
