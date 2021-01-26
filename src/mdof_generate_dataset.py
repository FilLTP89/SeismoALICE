u'''General informations'''
__author__ = "Luca Rosafalco"
__email__ = "luca.rosafalco@polimi.it"
r"""Select waveforms and simulate building response calling maatlab redbkit"""
u'''Required modules'''
import os
from mdof_data_selection import data_selector
from mdof_data_selection import data_selector_with_plot
import h5py
import pandas
import matlab.engine
import numpy as np
import pandas as pd
import scipy
from scipy.signal import detrend, windows

updated_STEAD = 0
nsy           = 10
ntm           = 4096
dire          = 0   # dire={0,1,2} stays for {x,y,z}

# data for the matlab routine #############################################################################################################
data_file_undamaged = 'telaio_stead_und'
data_file_damaged   = 'telaio_stead_dam'
# end data for the matlab routine #########################################################################################################

# data for the accelerograms ##############################################################################################################
# 'csv_file' is the name (.csv) of the considered chunk of the STandford EArthquake Dataset (STEAD) (https://github.com/smousavi05/STEAD)
# 'file_name' is the name (.hdf5) of the considered chunk of the STandford EArthquake Dataset (STEAD) (https://github.com/smousavi05/STEAD)
if updated_STEAD:
    earth_data_root = 'D:\\Luca\\Database\\STEAD\\new\\'
    csv_file        = 'chunk2.csv'
    file_name       = 'chunk2.hdf5'
else:
    earth_data_root = 'D:\\Luca\\Database\\STEAD\\'
    csv_file        = 'metadata_11_13_19.csv'
    file_name       = 'waveforms_11_13_19.hdf5'

csv_file        = earth_data_root + csv_file
file_name       = earth_data_root + file_name
# end data for the accelerograms ##########################################################################################################

# si immagina di dare in input anche un set di parametri capaci di effettuare i sampling
# STEAD data uploading ####################################################################################################################
if updated_STEAD:
    acc = data_selector(csv_file,file_name)
    acc = np.transpose(acc)
else:
    acc = -999.9*np.ones(shape=(nsy,ntm))
    acc_set = -999.9*np.ones(shape=(nsy,3,ntm))
    pgat_set = -999.9*np.ones(shape=(nsy,3))
    eqd = h5py.File(file_name,'r')['earthquake']['local']
    eqm = pd.read_csv(csv_file)
    eqm = eqm.loc[eqm['trace_category'] == 'earthquake_local']
    eqm = eqm.loc[eqm['source_magnitude'] >= 3.5]
    eqm = eqm.sample(frac=nsy/len(eqm)).reset_index(drop=True)
    w = windows.tukey(ntm,5/100)
    for i in eqm.index:
        tn = eqm.loc[i,'trace_name']
        bi = int(eqd[tn].attrs['p_arrival_sample'])
        for j in range(3):
            acc_set[i,j,:] = detrend(eqd[tn][bi:bi+ntm,j])*w
            pgat_set[i,j] = np.abs(acc_set[i,j,:]).max()
            acc_set[i,j,:] = acc_set[i,j,:]/pgat_set[i,j]
            pgat_set[i,j] = np.abs(acc_set[i,j,:]).max()
    #         _,psa,_,_,_ = rsp(md['dtm'],acc_set[i,j,:],md['vTn'],5.)
    #         psat_set[i,j,:] = psa.reshape((md['nTn']))
    acc = acc_set[:,dire,:]
    del eqd
# end STEAD data uploading ################################################################################################################

# data generator part #####################################################################################################################
# undamaged part
eng = matlab.engine.start_matlab()
acc_mat =  acc.tolist()
acc_mat = matlab.double(acc_mat)
# acc_mat =  acc.tolist()
eng.telaio_2_for_py(data_file_undamaged,acc_mat,nargout=0)

# damaged part
eng.telaio_2_for_py(data_file_damaged,acc_mat,nargout=0)
# end data generator part #################################################################################################################