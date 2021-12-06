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
from CommonNN import to_categorical
from CommonTorch import o1l, data_utils, rndp, Subset, _accumulate, dloader
from obspy import Stream, read, read_inventory
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
import GenerateNoise as gng
from GenerateNoise import filter_ths_tensor as flt
from GenerateNoise import lowpass_biquad
from GenerateNoise import filter_Gauss_noise as flg
from GenerateNoise import Gauss_noise as gn
from GenerateNoise import pollute_ths_tensor as pgn
from synthetic_generator import sp
from scipy.integrate import cumtrapz
from torch import FloatTensor as tFT
from ss_process import rsp
from scipy.signal import detrend, windows
from scipy.io import loadmat


rndm_args = {'mean': 0, 'std': 1}
dirs = {'ew':0,'ns':1,'ud':2}
u'''General informations'''

__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
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

    def __init__(self,inpX,inpY,tar,idx,*inpR):
        super(thsTensorData,self).__init__()

        self.inpX = inpX
        self.inpY = inpY
        # self.inpZ = inpZ
        # self.inpW = inpW
        self.tar  = tar
        self.idx  = idx
        self.inpR = inpR

    def __len__(self):
        return len(self.idx)
 
    def __getitem__(self,idx):
        X = self.inpX[self.idx[idx],:,:]
        Y = self.inpY[self.idx[idx],:,:]
        # Z = self.inpZ[self.idx[idx],:,:]
        # W = self.inpW[self.idx[idx],:,:]
        y = self.tar[self.idx[idx],:]
        # w = w[self.idx[idx],:]
        if hasattr(self, 'inpR'):
            try:
                R0 = self.inpR[0][self.idx[idx],:,:]
                R1 = self.inpR[1][self.idx[idx],:,:]
                R2 = self.inpR[2][self.idx[idx],:,:]
                return X, Y, y, R0, R1, R2
            except:
                return X, Y, y
        else:
            return X, Y, y

class HDF5thsTensorData(data_utils.Dataset):
    # https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    def __init__(self, file_path, recursive, load_data, 
        data_cache_size=3, groups = None, dsets=None, transform=None):
        super(HDF5thsTensorData,self).__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.groups = groups
        self.dsets = dsets
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds.value, file_path)

                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds.value.shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """

        with h5py.File(file_path) as h5_file:
            if not self.groups:
                groups = h5_file.items()
            for gname, group in groups:
                if not self.dsets:
                    dsets = group.items()
                for dname, ds in dsets:
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds.value, file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]


def GetSTEADDataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):

    vtm = md['dtm']*np.arange(0,md['ntm'])
    tar = np.zeros((nsy,2))
    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']))
    pgat_set = -999.9*np.ones(shape=(nsy,3))

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

    pgat_set = np2t(np.float32(pgat_set))
    trn_set = np2t(trn_set).float()

    ths_fsc = []

    partition = {'all': range(0,nsy)}
    trn = max(1,int(batch_percent[0]*nsy))
    tst = max(1,int(batch_percent[1]*nsy))
    vld = max(1,nsy-trn-tst)

    thf_set = lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])
    pgaf_set = -999.9*np.ones(shape=(nsy,3))

    for i in range(thf_set.shape[0]):
        for j in range(3):
            pgaf_set[i,j] = np.abs(thf_set[i,j,:].data.numpy()).max(axis=-1)

    pgat_set = np2t(np.float32(pgat_set))
    pgaf_set = np2t(np.float32(pgaf_set))

    tar[:,0] = eqm['source_magnitude'].to_numpy(np.float32)
    tar[:,1] = eqm['source_depth_km'].to_numpy(np.float32)

    nhe = tar.shape[1]
    tar_fsc = select_scaler(method='fit_transform',scaler='StandardScaler')
    tar = operator_map['fit_transform'](tar_fsc,tar)
    lab_enc = LabelEncoder()
    for i in range(nhe):
        tar[:,i] = lab_enc.fit_transform(tar[:,i])
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar = np2t(np.float32(one_hot.fit_transform(tar)))
    # plot_ohe(tar)
    fsc = {'lab_enc':lab_enc,'tar':tar_fsc,'ths_fsc':ths_fsc,\
           'one_hot':one_hot,'ncat':tar.shape[1]}
    ths = thsTensorData(trn_set,thf_set,tar,partition['all'])

    # RANDOM SPLIT
    idx,ths_trn = random_split(ths,[trn,vld,tst])
    ths_trn,ths_tst,ths_vld = ths_trn
    return ths_trn,ths_tst,ths_vld,vtm,fsc

def GetSTEADDatasetMPI(comm,size,rank,src,batch_percent,workers,
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

    vtm = md['dtm']*np.arange(0,md['ntm'])

    tar     = np.zeros((nsy,2))

    eqm = dd.read_csv(urlpath=src.replace('.hdf5','.csv').replace('waveforms','metadata'),
        na_values='None',dtype=meta).query('trace_category == "earthquake_local"').query('source_magnitude>=3.5')

    eqm = eqm.sample(frac=nsy/len(eqm),replace=True).reset_index(drop=True).set_index("trace_name",sorted=True)

    nsy_set = len(eqm)//size

    nsy = nsy_set*size

    trn_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']),dtype=np.float32)
    thf_set  = -999.9*np.ones(shape=(nsy,3,md['ntm']),dtype=np.float32)

    pgat_set = -999.9*np.ones(shape=(nsy,3),dtype=np.float32)
    pgaf_set = -999.9*np.ones(shape=(nsy,3),dtype=np.float32)

    trn_set  = -999.9*np.ones(shape=(nsy_set,3,md['ntm']),dtype=np.float32)
    thf_set  = -999.9*np.ones(shape=(nsy_set,3,md['ntm']),dtype=np.float32)

    pgat_set = -999.9*np.ones(shape=(nsy_set,3),dtype=np.float32)
    pgaf_set = -999.9*np.ones(shape=(nsy_set,3),dtype=np.float32)
    
    h5ls = H5ls(nsy=nsy_set,names=eqm.index.compute().to_list()[rank*nsy_set:(rank+1)*nsy_set],
        xw=Xwindow)

    f = h5py.File(src,'r',driver='mpio',comm=comm)
    dsets = f['earthquake']['local']
    # this will now visit all objects inside the hdf5 file and store datasets in h5ls.names
    dsets.visititems(h5ls)
    trn_set  = h5ls.ths
    pgat_set = h5ls.pga
    comm.barrier()
        # f.close()
    f.close()
    trn_set  = np2t(np.float32(trn_set))
    pgat_set = np2t(np.float32(pgat_set))

    thf_set = lowpass_biquad(trn_set,1./md['dtm'],md['cutoff'])

    for i in range(thf_set.shape[0]):
        for j in range(3):
            pgaf_set[i,j] = np.abs(thf_set[i,j,:].data.numpy()).max(axis=-1)

    pgat_set = np2t(np.float32(pgat_set))
    pgaf_set = np2t(np.float32(pgaf_set))

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
    one_hot = MultiLabelBinarizer(classes=range(64))
    tar_set = np2t(np.float32(one_hot.fit_transform(tar_set)))

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

        ths = thsTensorData(trn_set,thf_set,tar,partition['all'])

        # RANDOM SPLIT
        idx,ths_trn = random_split(ths,[trn,vld,tst])
        ths_trn,ths_tst,ths_vld = ths_trn
        return ths_trn,ths_tst,ths_vld,vtm,fsc


# Load mseed datasets downloaded via FDSN 
def LoadMSEEDDataset(source,inventory):
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

def GetANN2BBDataset(src,batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    
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
     
def GetSabettaPuglieseDataset(batch_percent,Xwindow,zwindow,nzd,nzf,md,nsy,device):
    
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

def GetMSEEDDataset(batch_percent,source=['./signals/*.*.*.mseed'],\
                 inventory=['./signals/sxml/RM07.xml',
                            './signals/sxml/LXRA.xml',
                            './signals/sxml/SRN.xml']):
    u'''LOAD DATASET'''
    inv,stt = LoadMSEEDDataset(source,inventory)    
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
