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

class EncoderDataParallele(object):
    """docstring for EncoderDataParallele"""
    def __init__(self, *arg, **kwargs):
        super(EncoderDataParallele, self).__init__()
        pass
        
    @staticmethod
    def getEncoder(name, ngpu, dev, nz, nch, ndf, nly, ker, std, pad, dil, *args, **kwargs):
        
        if name is not None:
            classname = 'Encoder_'+name
            try:
                return type(classname, (BasicEncoderDataParallele, ), dict(ngpu = ngpu,dev =dev, nz = nz, nch = nch, 
                        nly = nly, ker = ker, std =std, pad = pad, dil = dil))
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return Encoder(ngpu=ngpu, dev = dev,nz = nz, nch = nch, ndf = ndf,\
                nly=nly,ker=ker,std=std,pad=pad, dil=dil,  *args, **kwargs)

class BasicEncoderDataParallele(Module):
    """docstring for BasicEncoderDataParallele"""
    def __init__(self):
        super(BasicEncoderDataParallele, self).__init__()
        self.training = True

    def lout(self,nz,nch, nly, increment):
        """
        This code is for conv1d made according to the rule of Pytorch.
        One multiply nz by 2 ^(increment - 1). 
        If, by example, nly 8. we strat from nz^(0) to nz^(6). we stop witnz
        
        """
        n = increment - 1
        return int(nz*2**n) if n <= (nly - 2) else nz
    
class Encoder(BasicEncoderDataParallele):
    """docstring for Encoder"""
    def __init__(self, ngpu,dev,nz,nch,ndf,\
                 nly,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder, self).__init__()
        self.ngpu= ngpu
        act = [LeakyReLU(1.0,True) for i in range(1, nly+1)]
        self.gang = range(self.ngpu)
        in_channels = nz
        for i in range(1, nly+1):
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn = cnn1d(nch*2,ndf,act[0],ker=ker,std=std,pad=pad,dil=dil, bn=bn,dpc=0.0,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn = cnn1d(nch*2,ndf,act[0],ker=ker,std=std,pad=pad,dil=dil,bn=bn,dpc=0.0,wn=True)
            #else we proceed normaly
            else:
                out_channels = self.lout(nz,nch,nly,i)
                self.cnn += cnn1d(in_channels,out_channels, act[i-1], ker=ker,std=std,pad=pad,dil=dil,\
                        bn=False,dpc=0.0,wn=False) 
            in_channels = out_channels
        self.cnn = sqn(*self.cnn)

    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,x,self.gang)
            torch.cuda.empty_cache()
        else:
            zlf   = self.cnn(x)
        if not self.training:
            zlf =zlf.detach()
        return zlf