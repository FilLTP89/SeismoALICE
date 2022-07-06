import os
import math
import torch
from torch import from_numpy
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift
from configuration import app
from torch.utils.data import Dataset

class Toyset(Dataset): 
    def __init__(self, time=40.96, delta_t = 0.01, nsy = 1280,
                num_channels  = 3,latent_space  = [64,64],latent_channel= [16,32]):
        #time =
        app.NOISE = {'mean': 0, 'std': 1}
        N = round(time/delta_t)
        t = np.linspace(0.,time, N)
        omega = 2*math.pi/time
        
        # breakpoint()
        sig  = sinus()
        sig_noise = signal(_noise=True)

        # signal = (signal - np.mean(signal))/np.std(signal)

        self._dataset_x = torch.empty(*[nsy,3,N])
        self._dataset_y = torch.empty(*[nsy,3,N])
        # self._latent_space_x = torch.empty(*[nsy,latent_channel[0],latent_space[0]])
        self._latent_space_x = torch.empty(*[nsy,latent_channel[0]])
        self._latent_space_x = self._latent_space_x.normal_(app.NOISE['mean'],app.NOISE['std'])

        # self._latent_space_y = torch.empty(*[nsy,latent_channel[1],latent_space[1]])
        self._latent_space_y = torch.empty(*[nsy,latent_channel[1]])
        self._latent_space_y = self._latent_space_y.normal_(app.NOISE['mean'],app.NOISE['std'])

        # for broadband
        for n in range(nsy):
            for c in range(num_channels):
                self._dataset_y[n,c,:] = from_numpy(signal(_noise=True))
        # filtered
        for n in range(nsy):
            for c in range(num_channels):
                self._dataset_x[n,c,:] = from_numpy(sinus())

    def __len__(self): 
        return len(self._dataset_y)

    def __getitem__(self,index):
        return (self._dataset_y[index,:,:],
                    self._dataset_x[index,:,:],\
                    self._latent_space_y[index,:,],\
                    self._latent_space_x[index,:,])

"""
    Defining here some toy signal, for which we clearly know the equations.
"""
def signal(time = 40.96,delta_t = 0.01, _noise = False, drawing = False):
    # number of time step
    N     = round(time/delta_t)
    # discretization of time
    t     = np.linspace(0.,time, N)
    # angle of the signals
    omega = 2*math.pi/time
    # adding different frequencies of the signals
    n_t   = np.arange(5)
    signal= 0
    for i in range(len(n_t)):
        signal += np.sin(omega*t*n_t[i])
    # generating a random noise between [-1, +1] if needed
    noise  =  0  if _noise ==  False else np.random.normal(0.,1.0,len(t))
    # finally ...
    signal =  signal + noise

    #normalization of the signals betwenn [-1,1]
    signal = normalization(signal)

    if drawing == True:
        return t, signal
    else:
        return signal

def sinus(time = 40.96, delta_t = 0.01, _noise = False, drawing = False):
    # number of time steps
    np.random.seed(100)
    N = round(time/delta_t)
    # discretization of the time steps 
    t = np.linspace(0.,time, N)
    omega = 2*math.pi/time

    signal = np.sin(omega*t)
    noise  =  0  if _noise ==  False else np.random.normal(0.,1.0,len(t))
    signal = signal + noise

    if drawing == True :
        return t, normalization(signal)
    else:
        return normalization(signal)

def normalization(signal): 
    return -1 + 2*(signal -np.min(signal))/(np.max(signal) - np.min(signal))

def torch_to_numpy(x): 
    return x.cpu().data.numpy().copy()

def filtered(x):
    x = torch_to_numpy(x)
    b, a = signal.ellips(4, 0.01, 120, 0.125)
    fgust = signal.filtfilt(b, a, x, method="gust")
    return fgust

def get_dataset(dataset, nsy = 64, batch_size = 64, rank = 0, world_size = 1):
    dataset = Toyset(nsy = nsy)
    batch_size = batch_size
    train_part = int(0.80*len(dataset))
    vld_part   = len(dataset) - train_part
    
    train_set, vld_set = torch.utils.data.random_split(dataset, [train_part,vld_part])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank)

    vld_sampler = torch.utils.data.distributed.DistributedSampler(
            vld_set,
            num_replicas=world_size,
            rank=rank)

    trn_loader = torch.utils.data.DataLoader(dataset=train_set, 
                batch_size =batch_size, 
                shuffle    =False,
                num_workers=0,
                pin_memory =True,
                sampler    = train_sampler)

    vld_loader = torch.utils.data.DataLoader(dataset=vld_set, 
                batch_size =batch_size, 
                shuffle    =False,
                num_workers=0,
                pin_memory =True,
                sampler    = vld_sampler)
    return trn_loader, vld_loader

