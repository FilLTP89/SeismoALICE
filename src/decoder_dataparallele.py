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

class DecoderDataParallele(object):
    """docstring for DecoderDataParallele"""
    def __init__(self,*arg, **kwargs):
        super(DecoderDataParallele, self).__init__()
        pass
    
    @staticmethod
    def getDecoder(name,ngpu, nz, nch, nly, ndf, ker, std, pad, opd, dpc, limit):
        if name  is not None:
            classname = 'Decoder_'+ name
            try:
                return type(classname, (BasiceDecoderDataParallele, ), dict(ngpu = ngpu, nz = nz, nch = nch, nly = nly,\
                 ker = ker, std =std, pad = pad, limit=limit))
            except Exception as e:
                raise e
                print("The class ",classname," does not exit")
        else:
            return Decoder(ngpu,nz,nch,ndf,nly,\
                 ker=ker[i-1],std=std[i-1],pad=pad[i-1],opd=opd[i-1],dil=1,grp=1,dpc=dpc,limit=limit)


class BasiceDecoderDataParallele(Module):
    """docstring for BasiceDecoderDataParallele"""
    def __init__(self):
        super(BasiceDecoderDataParallele, self).__init__()
        self.training = True
    
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
    def __init__(self,ngpu,nz,nch,ndf,nly,channel,\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,limit = 256):
        super(Decoder, self).__init__()
        self.ngpu= ngpu
        self.gang = range(self.ngpu)
        act = [ReLU(inplace=True) for t in range(nly-1)]+[Tanh()]
        in_channels   = nz

        for i in range(1, nly+1):
            
            """
            This is made in the respectful of the pytorch documentation  :
            See the reference : 
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

            """
            # if we strat we initialize the cnn
            if i == 1:
                self.cnn = cnn1dt(channel[i-1],channel[i], act[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],opd=opd[i-1],bn=True,dpc=dpc)
            #else we conitnue adding
            else:  
                 self.cnn += cnn1dt(channel[i-1],channel[i], act[i-1],ker=ker[i-1],\
                    std=std[i-1],pad=pad[i-1],opd=opd[i-1], bn=False,dpc=0.0)
        self.cnn = sqn(*self.cnn)


    def forward(self,zxn):
        if zxn.is_cuda and self.ngpu > 1:
            Xr = pll(self.cnn,zxn,self.gang)
            torch.cuda.empty_cache()
        else:
            Xr = self.cnn(zxn)
        if not self.training:
            Xr=Xr.detach()
        return Xr
    
    
