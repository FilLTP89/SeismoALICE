import torch
import torch.nn as nn
from functools import partial


class BasicModel(nn.Module):  
    def __init__(self, model_name=None, *args, **kwargs):
        super(BasicModel, self).__init__()
        self.model_name = model_name
        
    def normalization_type(self,type='batch'):
        type_norm={
                    "batch"     : partial(nn.BatchNorm1d), 
                    "instance"  : partial(nn.InstanceNorm1d),
                    "spectral"  : partial(nn.utils.spectral_norm),
                    "layer"     : partial(nn.LayerNorm)
                }
        norm = type_norm.get(type, partial(nn.BatchNorm1d))
        return norm

    @property
    def number_parameter(self):
        return sum(p.numel() for p in self.parameters())