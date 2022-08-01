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

class MixedGaussianDataset(Dataset):
    """ MixedGaussianDistributionDataset
        This class is created to evaluate if the discriminator are able to distinguish between 
        N(0,I) and another  gaussian distribution. A classifier logic training on the discriminator should be used to
        test wether or not the discriminator is powerful enough
    """
    def __init__(self,latent_space_shape=[[1,512],[1, 512]],nsy = 1280, mean=0., 
                        std = 1.0, seed=None,*args, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        self._latent_space_gaussian_normal = torch.empty(*[nsy,*latent_space_shape[0]]).normal_(mean=0.,std=1.0)
        self._latent_space_distribution = torch.empty(*[nsy,*latent_space_shape[0]]).normal_(mean=mean,std=std)

    def __len__(self): 
        return len(self._latent_space_gaussian_normal)

    def __getitem__(self,index):
        return (self._latent_space_gaussian_normal[index,:],self._latent_space_distribution[index,:])

class MixedGaussianUniformDataset(Dataset):
    """ MixedGaussianUniformDataset
        This class is created to evaluate, as the MixedGaussianDistriution class, wether the discriminator
        could distinguish between a gaussan distribution and another kind of distribution that is not gaussian.
        It the discriminator is not powerful enough to make that distinction, somthing skeewed up in 
        the architecture of the discriminator
    """
    def __init__(self,latent_space_shape=[[1,512],[1, 512]],nsy = 1280,start=0., 
                        end = 1.0, seed=None,*args, **kwargs):
        if seed is not None:
            torch.manual_seed(seed)
        self._latent_space_gaussian_normal  = torch.empty(*[nsy,*latent_space_shape[0]]).normal_(mean=0.,std=1.0)
        self._latent_space_distribution     = torch.empty(*[nsy,*latent_space_shape[0]]).uniform_(start,end)

    def __len__(self): 
        return len(self._latent_space_gaussian_normal)

    def __getitem__(self,index):
        return (self._latent_space_gaussian_normal[index,:],self._latent_space_distribution[index,:])

def get_latent_dataset(dataset = LatentDataset, nsy=1280, batch_size=64, *args, **kwargs):
    _dataset    = dataset(nsy=nsy,*args,**kwargs)
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

