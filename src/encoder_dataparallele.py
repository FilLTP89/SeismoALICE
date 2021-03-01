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
    def getEncoder(name, ngpu, dev, nz, nch, ndf, nly, ker, std, pad, dil, act,limit, *args, **kwargs):
        
        if name is not None:
            classname = 'Encoder_'+name
            try:
                return type(classname, (BasicEncoderDataParallele, ), dict(ngpu = ngpu,dev =dev, nz = nz, nch = nch, act=act,
                        nly = nly, ker = ker, std =std, pad = pad, dil = dil, limit = limit))
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

    def lout(self,nz,nch, nly, increment,limit):
        """
        This code is for conv1d made according to the rule of Pytorch.
        One multiply nz by 2 ^(increment - 1). 
        If, by example, nly 8. we strat from nz^(0) to nz^(6). we stop witnz
        
        """
        limit = 256
        n = increment - 1
        val = int(nz*2**n) if n <= (nly - 2) else nz
        return val if val <= limit else limit
    
class Encoder(BasicEncoderDataParallele):
    """docstring for Encoder"""
    def __init__(self, ngpu,dev,nz,nch,ndf,act,\
                 nly,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256,\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder, self).__init__()
        self.ngpu= ngpu
        act = [LeakyReLU(1.0,True) for i in range(1, nly+1)]
        self.gang = range(self.ngpu)
        
        for i in range(1, nly+1):
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn = cnn1d(nch*2,ndf,act[0],ker=ker[i-1][i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1], bn=bn,dpc=dpc,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn = cnn1d(nch*2,ndf,act[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1],bn=bn,dpc=dpc,wn=True)
            #else we proceed normaly
            else:
                
                _bn = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc
                self.cnn += cnn1d(channel[i-1],channel[i], act[i-1], ker=ker[i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1],\
                        bn=_bn,dpc=_dpc,wn=False) 
            in_channels = out_channels
        self.cnn = sqn(*self.cnn)

    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,x,self.gang)
        else:
            zlf   = self.cnn(x)
        if not self.training:
            zlf =zlf.detach()
        torch.cuda.empty_cache()
        return zlf