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
import copy

class DecoderDataParallele(object):
    """docstring for DecoderDataParallele"""
    def __init__(self,*arg, **kwargs):
        super(DecoderDataParallele, self).__init__()
        pass
    
    @staticmethod
    def getDecoder(name,ngpu, nz, nch, nly, ndf, ker, std, pad, dil, channel,n_extra_layers,act, opd, bn, dpc,path, limit):
        if name  is not None:
            classname = 'Decoder_'+ name
            try:
                return type(classname, (BasiceDecoderDataParallele, ), dict(ngpu = ngpu, nz = nz, nch = nch, nly = nly,\
                 ker = ker, std =std, pad = pad, limit=limit, path=path,channel = channel,n_extra_layers=n_extra_layers,opd=opd, dil=dil, act=act, bn=bn))
            except Exception as e:
                raise e
                print("The class ",classname," does not exit")
        else:
            return Decoder(ngpu = ngpu, nz = nz, nch = nch, limit = limit, bn = bn, path=path,\
        nly = nly, act=act, ndf =ndf, ker = ker, std =std, pad = pad, opd = opd,\
        dil=dil, dpc = dpc,n_extra_layers=n_extra_layers, channel = channel)


class BasiceDecoderDataParallele(Module):
    """docstring for BasiceDecoderDataParallele"""
    def __init__(self):
        super(BasiceDecoderDataParallele, self).__init__()
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

class Decoder(BasiceDecoderDataParallele):
    """docstring for Decoder"""
    def __init__(self,ngpu,nz,nch,ndf,nly,channel,act,\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,limit = 256, bn=True, path='',n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu= ngpu
        self.gang = range(self.ngpu)
        acts       = T.activation(act, nly)
        device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        # pdb.set_trace()
        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False
        # pdb.set_trace()
        for i in range(1, nly+1):
            """
            This is made in the respectful of the pytorch documentation  :
            See the reference : 
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

            """
            _dpc = 0.0 if i ==nly else dpc
            _bn =  False if i == nly else bn
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1], opd=opd[i-1], bn=_bn,dpc=_dpc)
            

        for i in range(0,n_extra_layers):
            #adding LeakyReLU activation function
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[0],ker=3,std=1,pad=1,\
                dil =1, opd=0, bn=True, dpc=0.0)

        # pdb.set_trace()
        self.cnn1 = sqn(*self.cnn1)
        if path: 
            self.cnn1[-1] = self.model
        self.cnn1.to(device)


    def forward(self,zxn):
        if zxn.is_cuda and self.ngpu > 1:
            Xr = pll(self.cnn1,zxn,self.gang)
            # Xr = T._forward(zxn, self.cnn1, self.gang)
            # torch.cuda.empty_cache()
        else:
            Xr = self.cnn1(zxn)
        if not self.training:
            Xr=Xr.detach()
        torch.cuda.empty_cache()
        return Xr
    
    
