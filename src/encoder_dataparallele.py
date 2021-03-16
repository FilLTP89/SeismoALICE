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

class EncoderDataParallele(object):
    """docstring for EncoderDataParallele"""
    def __init__(self, *arg, **kwargs):
        super(EncoderDataParallele, self).__init__()
        pass
        
    @staticmethod
    def getEncoder(name, ngpu, dev, nz, nch, ndf, nly, ker, std,\
                pad, dil, channel, act,limit,path, *args, **kwargs):
        
        if name is not None:
            classname = 'Encoder_'+name
            try:
                return type(classname, (BasicEncoderDataParallele, ), dict(ngpu = ngpu,dev =dev, nz = nz, nch = nch, act=act,
                        nly = nly, ker = ker, ndf=ndf,std =std, path=path, pad = pad, dil = dil, channel=channel, limit = limit))
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return Encoder(ngpu = ngpu,dev =dev, ndf=ndf, nz = nz, nch = nch, act=act,
                        nly = nly, ker = ker, std =std, pad = pad, path=path, dil = dil, channel=channel, limit = limit)

class BasicEncoderDataParallele(Module):
    """docstring for BasicEncoderDataParallele"""
    def __init__(self):
        super(BasicEncoderDataParallele, self).__init__()
        self.training = True
        self.model = None
        self.cnn1 = []
        

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
    def __init__(self, ngpu,dev,nz,nch,ndf,act, channel,\
                 nly,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256, path='',\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder, self).__init__()
        self.ngpu= ngpu
        self.gang = range(ngpu)
        
        device = tdev("cuda" if torch.cuda.is_available() else "cpu")

        acts = T.activation(act, nly)
        # pdb.set_trace()

        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False
        # pdb.set_trace()
        # lin = 4096
        for i in range(1, nly+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            # lin = lout
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 += cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil = dil[i-1], dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 += cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                _bn  = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc 
                self.cnn1 += cnn1d(channel[i-1], channel[i], acts[i-1], ker=ker[i-1],\
                    std=std[i-1],pad=pad[i-1], dil=dil[i-1], bn=_bn, dpc=_dpc, wn=False)
        
        self.cnn1  = sqn(*self.cnn1)
        # pdb.set_trace()
        if path:
            self.model.cnn1[-1] = copy.deepcopy(self.cnn1)
            self.cnn1 = self.model
        self.cnn1.to(device)

    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            # zlf   = pll(self.cnn,x,self.gang)
            x = x.to()
            zlf = T._forward(x, self.cnn1, self.gang)
        else:
            zlf   = self.cnn1(x)
        if not self.training:
            zlf =zlf.detach()
        torch.cuda.empty_cache()
        return zlf