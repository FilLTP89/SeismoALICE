from torch.nn.modules import activation

u'''AE design'''
from torch.nn.modules import activation
u'''AE design'''
u'''Required modules'''
import warnings
# import GPUtil
warnings.filterwarnings("ignore")
from common_nn import *
import torch
import pdb
from torch import device as tdev
import importlib
import copy


class Base(Module):
    def __init__(self):
        super(Base,self).__init__()
        # interpeting data from config value
        self.padding = T.interpreter(config,"padding")
        self.kernel  = T.interpreter(config,"kernel")
        self.strides = T.interpreter(config,"strides")
        self.nly     = T.interpreter(config,"nlayer")
        self.dilation= T.interpreter(config,"dilation")
        self.acts    = T.activation(T.interpreter(config,"act"))
        self.nly     = T.interpreter(config,"nly")
        self.channel = T.interpreter(config,"channel")
        self.dpc     = T.interpreter(config,"dpc")
        self.limit   = T.interpreter(config,"limit")
        self.wf      = T.interpreter(config,"wf")
        self.dpc     = T.interpreter(config,"dpc")

    def forward(self, x): 
        pass