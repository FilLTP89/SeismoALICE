import torch
import torch.nn as nn
from functools import partial


class BasicModel(nn.Module):  
    def __init__(self, model_name=None, *args, **kwargs):
        super(BasicModel, self).__init__()
        self.model_name = model_name
        
    def layer_output_normalization_type(self,type='batch'):
        type_norm={
                    "batch"     : partial(nn.BatchNorm1d), 
                    "instance"  : partial(nn.InstanceNorm1d),
                    "layer"     : partial(nn.LayerNorm)
                }
        norm = type_norm.get(type, partial(nn.BatchNorm1d))
        return norm

    def weight_regularization_type(self,type):
        type_regularization={
                    "spectral"   : partial(nn.utils.spectral_norm),
                    "orthogonal" : partial(nn.utils.parametrizations.orthogonal)
                }
        regularization = type_regularization.get(type, partial(nn.utils.spectral_norm))
        return regularization

    @property
    def number_parameter(self):
        return sum(p.numel() for p in self.parameters())