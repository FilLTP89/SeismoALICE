import os
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from configuration import app
from torch.utils.data import Dataset

class LatentDataset(Dataset): 
    def __init__(self, latent_space_shape=[[1,1024],[4, 1024]],nsy = 1280, mean=0., 
                        std = 1.0, seed=123,*args, **kwargs):
        # fix the seed for the same generation of gaussian, independently of epoch
        torch.manual_seed(seed)
        self._latent_space_zhf = torch.randn(*[nsy,*latent_space_shape[0]])
        self._latent_space_zlf = torch.randn(*[nsy,*latent_space_shape[1]])
        
    def __len__(self): 
        return len(self._latent_space_zhf)

    def __getitem__(self,index):
        return (self._latent_space_zhf[index,:],self._latent_space_zlf[index,:])

class MultVariate(Dataset):
    def __init__(self,*args,**kwargs):
        pass
    


def get_latent_dataset(nsy=1280, batch_size=64, *args, **kwargs):
    dataset    = LatentDataset(nsy=nsy,*args,**kwargs)
    train_part,vld_part,tst_part = int(0.80*len(dataset)),int(0.10*len(dataset)),int(0.10*len(dataset))
    train_set, vld_set, tst_set  = torch.utils.data.random_split(dataset, [train_part,vld_part,tst_part])

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

