import os
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from configuration import app
from torch.utils.data import Dataset

class LatentDataset(Dataset): 
    def __init__(self, latent_space_shape=[[1,512],[4, 1024]],nsy = 1280, mean=0., 
                    std = 1.0, seed=None,*args, **kwargs):
        # fix the seed for the same generation of gaussian, independently of epoch
        if seed is not None:
            torch.manual_seed(seed)
        self._latent_space_zhf = torch.randn(*[nsy,*latent_space_shape[0]])
        self._latent_space_zlf = torch.randn(*[nsy,*latent_space_shape[1]])
        
    def __len__(self): 
        return len(self._latent_space_zhf)

    def __getitem__(self,index):
        return (self._latent_space_zhf[index,:],self._latent_space_zlf[index,:])

class UniformLatentDatatset(Dataset):
    def __init__(self,latent_space_shape=[[1,512],[4, 1024]],nsy = 1280, mean=0., 
                        std = 1.0, seed=None,*args, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        
        self._latent_space_zhf = torch.empty(*[nsy,*latent_space_shape[0]]).uniform_(1,5)
        self._latent_space_zlf = torch.empty(*[nsy,*latent_space_shape[0]]).uniform_(1,5)
    
    def __len__(self): 
        return len(self._latent_space_zhf)

    def __getitem__(self,index):
        return (self._latent_space_zhf[index,:],self._latent_space_zlf[index,:])

class GaussianDataset(Dataset):
    def __init__(self,latent_space_shape=[[1,512],[4, 1024]],nsy = 1280, mean=0., 
                        std = 1.0, seed=None,*args, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        
        self._latent_space_zhf = torch.empty(*[nsy,*latent_space_shape[0]]).normal_(mean=mean,std=std)
        self._latent_space_zlf = torch.empty(*[nsy,*latent_space_shape[0]]).normal_(mean=mean,std=std)
        
    def __len__(self): 
        return len(self._latent_space_zhf)

    def __getitem__(self,index):
        return (self._latent_space_zhf[index,:],self._latent_space_zlf[index,:])

def get_latent_dataset(dataset = LatentDataset, nsy=1280, batch_size=64, *args, **kwargs):
    _dataset    = LatentDataset(nsy=nsy,*args,**kwargs)
    train_part,vld_part,tst_part = int(0.80*len(_dataset)),int(0.10*len(_dataset)),int(0.10*len(_dataset))
    train_set, vld_set, tst_set  = torch.utils.data.random_split(_dataset, [train_part,vld_part,tst_part])

    trn_loader = torch.utils.data.DataLoader(dataset=train_set, 
                batch_size =batch_size, shuffle =True,
                num_workers=0, pin_memory =True)
    vld_loader = torch.utils.data.DataLoader(dataset=vld_set, 
                batch_size =batch_size, shuffle =True,
                num_workers=0,pin_memory =True)
    tst_loader = torch.utils.data.DataLoader(dataset=tst_set, 
                batch_size =batch_size, shuffle =True,
                num_workers=0,pin_memory =True)

    return trn_loader, vld_loader, tst_loader

