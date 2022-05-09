import torch
import torch.nn as nn


class BasicModel(nn.Module):  
    def __init__(self, model_name=None, *args, **kwargs):
        super(BasicModel, self).__init__()
        self.model_name = model_name
        

    @property
    def number_parameter(self):
        return sum(p.numel() for p in self.parameters())