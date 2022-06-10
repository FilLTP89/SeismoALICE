import os
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from configuration import app
from torch.utils.data import Dataset

class LatentDataset(Dataset): 
    def __init__(self, latent_space_shape=[[1,128],[3, 128]],nsy = 1280, mean=0., 
                        std = 1.0, seed=123,*args, **kwargs):
        # fix the seed for the same generation of gaussian, independently of epoch
        self._latent_space_zlf = torch.empty(*[nsy,*latent_space_shape[0]])
        self._latent_space_zlf = self._latent_space_zlf.normal_(mean, std)

        self._latent_space_zhf = torch.empty(*[nsy,*latent_space_shape[1]])
        self._latent_space_zhf = self._latent_space_zhf.normal_(mean, std)

    def __len__(self): 
        return len(self._latent_space_zlf)

    def __getitem__(self,index):
        return (self._latent_space_zlf[index,:],self._latent_space_zhf[index,:])

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

