import torch
import torch.nn as nn


class BasicModel(nn.Module):  
    def __init__(self):
        super(BasicModel, self).__init__()
        

    @property
    def number_parameter(self):
        return sum(p.numel() for p in self.parameters())