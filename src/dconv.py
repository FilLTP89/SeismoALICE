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
        self.channels  = []

        self.net = []

    def network(self):
        pass

class DConv_63(DConv):
    """docstring for DConv_63"""
    def __init__(self,conv,last_channel, bn = False, dpc = 0.0):
        super(DConv, self).__init__()

        #network parameters
        self.padding  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.dilation = [1,1,2,2,3,3,3,3,3,3,2,2,1,1,3,1,1]
        self.kernel   = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,1,1]
        self.stride   = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.channels = [last_channel,64,64,128,128,256,256,256,256,256,256,256,256,256,256,1024,1024,last_channel]
        self.net      = []
        #nomber of layers
        nly = len(self.kernel)
        # pdb.set_trace()
        for i in range(0, nly):
            # default activation function LeakyReLU
            act =  nn.ReLU(inplace=True) if i<= (nly -1) else nn.LeakyReLU(1.0,inplace=True)
            self.net += conv(in_channels = self.channels[i],\
                              out_channels = self.channels[i+1],\
                              ker = self.kernel[i],\
                              std = self.stride[i],\
                              pad = self.padding[i],\
                              bn  = bn,\
                              act = act,
                              dil = self.dilation[i],
                              dpc = dpc,
                              wn  = False)

    def network(self):
        # return convolutional network
        return self.net

class DConv_62(object):
  """docstring for DConv_12"""
  def __init__(self, conv,last_channel, bn = False, dpc = 0.0):
    super(DConv_62, self).__init__()
    self.kernel   = [3,3,3,3,3,3]
    self.stride   = [1,1,1,1,1,1]
    self.padding  = [0,0,0,0,0,0]
    self.dilation = [1,2,3,4,5,6]
    self.channels = [64,128,256,512,last_channel]

    #nomber of layers
    nly = len(self.kernel)
    # pdb.set_trace()
    for i in range(0, nly):
      # default activation function LeakyReLU
      act =  nn.ReLU(inplace=True) if i<= (nly -1) else nn.ReLU(inplace=True)
      self.net += conv(in_channels = self.channels[i],\
                            out_channels = self.channels[i+1],\
                            ker = self.kernel[i],\
                            std = self.stride[i],\
                            pad = self.padding[i],\
                            bn  = bn,\
                            act = act,
                            dil = self.dilation[i],
                            dpc = dpc,
                            wn  = False)

class Transopose_DConv_62(object):
  """docstring for DConv_12"""
  def __init__(self, conv,last_channel, bn = False, dpc = 0.0):
    super(Transopose_DConv_62, self).__init__()
    self.kernel   = [3,3,3,3,3,3,3]
    self.stride   = [1,1,1,1,1,1,1]
    self.padding  = [0,0,0,0,0,0,0]
    self.dilation = [2,3,4,5,6,6,5]
    self.outpads  = [0,0,0,0,0,0,0]
    self.channels = [64,128,256,512,512,last_channel]

    #nomber of layers
    nly = len(self.kernel)
    # pdb.set_trace()
    for i in range(0, nly):
      # default activation function LeakyReLU
      act =  nn.ReLU(inplace=True) if i<= (nly -1) else nn.Tanh()
      self.net += conv(in_channels = self.channels[i],\
                            out_channels = self.channels[i+1],\
                            ker = self.kernel[i],\
                            std = self.stride[i],\
                            pad = self.padding[i],\
                            bn  = bn,\
                            act = act,
                            dil = self.dilation[i],
                            dpc = dpc,
                            opd = self.outpads[i],
                            wn  = False)