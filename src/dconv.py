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


class DConv(object):
    def __init___(self):
        super(DConv, self).__init___()
        self.padding  = []
        self.dilation = []
        self.kernel   = []
        self.stride   = []
        self.channels = []
        self.net      = []

    def network(self):
        return self.net


class DConv_62(DConv):
    def __init__(self, last_channel, bn = False, dpc = 0.0):
        super(DConv_62, self).__init__()
        self.kernel   = [3,3,3,3,3,3]
        self.stride   = [1,1,1,1,1,1]
        self.padding  = [0,0,0,0,0,0]
        self.dilation = [1,2,3,4,5,6]
        self.channels = [32,128,256,512,1024,512,last_channel]
        self.net = []

        #nomber of layers
        nly = len(self.kernel)
        # pdb.set_trace()
        for i in range(0, nly):
          # default activation function LeakyReLU
            act =  nn.ReLU(inplace=True) if i<= (nly -1) else nn.ReLU(inplace=True)
            self.net += cnn1d(in_channels = self.channels[i],\
                out_channels = self.channels[i+1],ker = self.kernel[i],\
                std = self.stride[i], pad = self.padding[i],\
                bn  = bn,act = act,dil = self.dilation[i],dpc = dpc, wn  = False)


class Transpose_DConv_62(DConv):
    def __init__(self, last_channel, bn = True, dpc = 0.0):
        super(Transpose_DConv_62, self).__init__()
        self.kernel   = [3,3,3,3,3,3,3]
        nly = len(self.kernel)
        self.stride   = [1,1,1,1,1,1,1]
        self.padding  = [0,0,0,0,0,0,0]
        self.dilation = [2,3,4,5,6,6,5]
        self.outpads  = [0,0,0,0,0,0,0]
        self.channels = [last_channel]+[64,128,128,128,128,128]+[last_channel]
        self.net = []

        #nomber of layers
        
        # pdb.set_trace()
        for i in range(0, nly):
          # default activation function LeakyReLU
            act =  nn.ReLU(inplace=True) if i<= (nly-2) else nn.Tanh()
            _bn =  bn if i<= (nly-2) else not bn
            self.net += cnn1dt(in_channels = self.channels[i],out_channels = self.channels[i+1],\
                ker = self.kernel[i], std = self.stride[i],\
                pad = self.padding[i], bn  = _bn,\
                act = act, dil = self.dilation[i],dpc = dpc, opd = self.outpads[i])
