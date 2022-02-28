from torch.nn.modules import activation
from core.net.basic_model import BasicModel
from conv.resnet.residual import DecoderResnet,block_2x2
from torch.nn.modules import activation
from configuration import app
from common.common_nn import *
import torch
import pdb
from torch import device as tdev
import importlib
import copy
from conv.dconv.dconv import Transpose_DConv_62
from conv.oct_conv.oct_conv import OctaveBatchNormActivation
from conv.oct_conv.oct_conv import OctaveTransposedConv
from conv.resnet.convblock import ResidualBlock




class DecoderDataParallele(object):
    """docstring for DecoderDataParallele"""
    def __init__(self,*arg, **kwargs):
        super(DecoderDataParallele, self).__init__()
        pass
    
    @staticmethod
    def getDecoder(name, ngpu, nz, nch, nly, ndf, ker, std, pad, dil,dconv,\
                     channel,n_extra_layers,act, opd, bn, dpc,path, limit, extra):

        # pdb.set_trace()
        if name  is not None:
            classname = 'Decoder_'+ name
            module_name = "net.decoder.decoder_dataparallele"
            module = importlib.import_module(module_name)
            class_ = getattr(module,classname)
            try:
                return class_(ngpu = ngpu, nz = nz, nch = nch, limit = limit, bn = bn, path=path,\
                        nly = nly, act=act, ndf =ndf, ker = ker, std =std, pad = pad, opd = opd,dconv = dconv,\
                        dil=dil, dpc = dpc,n_extra_layers=n_extra_layers, channel = channel, extra = extra)
            except Exception as e:
                raise e
                print("The class ",classname," does not exit")
        else:
            return Decoder(ngpu = ngpu, nz = nz, nch = nch, limit = limit, bn = bn, path=path,\
        nly = nly, act=act, ndf =ndf, ker = ker, std =std, pad = pad, opd = opd,dconv = dconv,\
        dil=dil, dpc = dpc,n_extra_layers=n_extra_layers, channel = channel, extra = extra)


class BasicDecoderDataParallel(BasicModel):
    """docstring for BasicDecoderDataParallel"""
    def __init__(self):
        super(BasicDecoderDataParallel, self).__init__()
        self.training = True
        self.model    = None
        self.cnn1     = []

    def lout(self,nz, nch, nly, increment, limit):
        """
        This code is for convTranspose1d made according to the rule of Pytorch. See official reference :
        One  multiply nz by  2 ^ (nly -incremement -1). 
        if nly 5. we strate from nzd*2^(3), nzd*2^(2), nz*2^(1), nzd*2^(0), nch. 
        Here nch should be equal to 3. 
        Therefore the convolutionnal network strat form :
         (nz,nzd*2^(3)) --> (nzd*2^(3),nzd*2^(2)) --> (nzd*2^(2),nzd*2^(1))
         --> (nzd*2^(1),nzd*2^(0))--> (nzd*2^(0),nch)
        """
        nzd = nz
        n = nly-2-increment+1
        val = int(nzd*2**n) if n >=0 else nch
        return val if val<= limit else limit

    def forward(self,x):
        pass

    def feed(x,features):
        for layer, feature in zip(self.cnn1.children(),reverse(features)):
            x =  layer(x)
            if layer.__class__.__name__ == 'ConvTranspose1d' : 
                x = x + feature
        return x


class Decoder(BasicDecoderDataParallel):
    """docstring for Decoder"""
    def __init__(self,ngpu,nz,nch,ndf,nly,channel,act, extra, dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,limit = 256, bn=True, path='',n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        acts      = T.activation(act, nly)
        _dconv    = "" 
        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        # pdb.set_trace()
        # dconv = DConv_63(last_channel = channel[-1], bn = False, dpc = 0.0).network()
        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False
        
        if dconv:
            _dconv = Transpose_DConv_62(last_channel = channel[-1], bn = True, dpc = 0.0).network()

        # pdb.set_trace()
        for i in range(1, nly+1):
            _dpc = 0.0 if i ==nly else dpc
            _bn =  False if i == nly else bn
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1], opd=opd[i-1], bn=_bn,dpc=_dpc)
            
        # pdb.set_trace()
        for i in range(0,n_extra_layers):
            self.cnn1 += [ResidualBlock(channel[-1], activation_function='selu')]

        # pdb.set_trace()
        if dconv:
            self.cnn1  =  self.cnn1 + _dconv

        self.cnn1  = sqn(*self.cnn1)
        if path: 
            self.cnn1[-1] = self.model
        # self.cnn1.to(self.device)

    def forward(self,zxn, features = None):
        if features is None:
            Xr = self.cnn1(zxn)
            app.logger.debug("In Model: input size {} - {}  output size {} - {}".
                format(zxn.size(),zxn.device,Xr.size(),Xr.device))
        else:
            Xr = feed(zxn,features)
        if not self.training:
            result = Xr.detach() + 0.
            
            return result
        return Xr



class Decoder_Lite(BasicDecoderDataParallel):
    """docstring for Decoder"""
    def __init__(self,ngpu,nz,nch,ndf,nly,channel,act, extra, dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,limit = 256, bn=True, path='',n_extra_layers=0):
        super(Decoder_Lite, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        acts      = T.activation(act, nly)

        # pdb.set_trace()
        
        # self.cnn1 += [nn.Linear(limit, 10, bias=False)]
        # self.cnn1 += [nn.ReLU(inplace=True)]
        # self.cnn1 += [nn.Linear(10, channel[0]*extra, bias=False)]
        # self.cnn1 += [nn.ReLU(inplace=True)]
        # self.cnn1 += [Explode((channel[0],extra))]
        
        # self.cnn1 += [nn.Linear(limit,extra*limit)]
        # self.cnn1 += [nn.ReLU(inplace=True)]
        # self.cnn1 += [Explode(shape=(extra,limit))]

        for i in range(1, nly+1):
            _dpc = 0.0 if i ==nly else dpc
            _bn  = False if i == nly else bn
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],
                ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1], bn=_bn,dpc=_dpc)
        # pdb.set_trace()
        self.cnn1  = sqn(*self.cnn1)

    def forward(self,zxn):
        # pdb.set_trace()
        Xr = self.cnn1(zxn)
        app.logger.debug("In Model: input size {} - {}  output size {} - {}".
                format(zxn.size(),zxn.device,Xr.size(),Xr.device))
        if not self.training:
            result = Xr.detach()
            return result
        return Xr
        
class Decoder_Resnet(BasicDecoderDataParallel):
    """docstring for Decoder_Resnet"""
    def __init__(self, ngpu,nz,nch,ndf,nly,channel,act, extra,dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,limit = 256,\
                 bn=True, path='',n_extra_layers=0):
        super(Decoder_Resnet, self).__init__()
        
        _net = DecoderResnet(in_signals_channels = channel[0],
                out_signals_channels=3,
                channels = [16, 32, 64], 
                layers = [2,2,2], block=block_2x2
            )
        self.net =  _net

    def forward(self,x):
        x = self.net(x)
        return x

class Decoder_Octave(BasicDecoderDataParallel):
    """docstring for Decoder_Octave"""
    def __init__(self,ngpu,nz,nch,ndf,nly,channel,act, extra,dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,limit = 256,\
                 bn=True, path='',n_extra_layers=0):
        super(Decoder_Octave, self).__init__()
        
        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")

        #defining the transposed octave convoluton layers 
        #activation functions are leakyReLU with a negative slope of 1.0
        self.layers = []
        self.leaky_relu =  T.activation_func("leaky_relu")
        for i in range(nly):
            self.layers += [OctaveBatchNormActivation(
                conv           = OctaveTransposedConv, 
                in_channels    = channel[i], 
                out_channels   = channel[i+1], 
                stride         = std[i],
                kernel_size    = ker[i], 
                padding        = pad[i], 
                dilation       = dil[i],
                activation_layer = act
            )]
        self.conv   = nn.Sequential(*self.layers)
        
        self.conv_h = nn.Sequential(*[
                nn.Identity(),
                self.leaky_relu
        ]).to(self.device)

        self.conv_l = nn.Sequential(*[nn.Conv1d(
                    in_channels = limit[0], 
                    out_channels = limit[2], 
                    stride = 2, 
                    kernel_size = 3,
                    padding = 1,
                    dilation = 1),
                self.leaky_relu
        ]).to(self.device)

    def forward(self,z):
        # x_h are High Frequency 
        # x_l are low Frequency
        x_h, x_l = self.conv(z)
        x_h = self.conv_h(x_h)
        x_l = self.conv_l(x_l)
        return x_h, x_l