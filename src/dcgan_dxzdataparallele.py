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


class DCGAN_DXZDataParallele(object):
    """docstring for DCGAN_DXZDataParallele"""
    def __init__(self, *args, **kwargs):
        super(DCGAN_DXZDataParallele, self).__init__()
        pass
        
    @staticmethod
    def getDCGAN_DXZDataParallele(name, ngpu, nly, channel, act, nc=1024,\
        ker=2,std=2,pad=0, dil=0,grp=0,limit = 256,\
        bn=True,wf=False, dpc=0.25, n_extra_layers= 0, bias = False, *args, **kwargs):

        if name is not None:
            classname = 'DCGAN_DXZ'+name
            #preparation for other DataParallele Class
            try:
                module_name = "dcgan_dxzdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)
                return class_(ngpu=ngpu, nc=nc, nly=nly, channel = channel,\
                                ker=ker,std=std,pad=pad, dil=dil, act = act,\
                                grp=grp, bn=bn, wf=wf, dpc=dpc,\
                                n_extra_layers=n_extra_layers, limit = limit, bias= bias)
            except Exception as e:
                raise e
                print("The class ", classname, " does not exit")
        else:
            return DCGAN_DXZ(ngpu=ngpu, nc=nc, nly=nly, channel = channel, act = act,\
                 ker=ker,std=std,pad=pad, dil=dil, grp=grp,\
                 bn=bn, wf=wf, dpc=dpc,\
                 n_extra_layers=n_extra_layers, limit = limit, bias=bias)

class BasicDCGAN_DXZDataParallele(Module):
    """docstring for BasicDCGAN_DXZDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DXZDataParallele, self).__init__()
        self.training = True
        
    def lout(self, nc, nly, increment, limit):
        val =  nc if increment < nly else 1
        return val if val <= limit else limit

    def critic(self,x):
        pass

    def extraction(self,x):
        pass

class DCGAN_DXZ(BasicDCGAN_DXZDataParallele):
    """docstring for DCGAN_DXZ"""
    def __init__(self,ngpu, nly, channel, act, nc=1024,\
        ker=2,std=2,pad=0, dil=1,grp=1,\
        bn=True,wf=False, dpc=0.25, limit =1024,\
        n_extra_layers= 0, bias=False, *args, **kwargs):
        super(DCGAN_DXZ, self).__init__()
        self.ngpu =  ngpu
        self.gang = range(self.ngpu)
        self.cnn  = []
        self.exf  = []

        #activation functions
        activation = T.activation(act,nly)

        #initialisation of the input channel
        in_channels = nc

        for i in range(1, nly+1):
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            
            # self.cnn += cnn1d(in_channels, out_channels, activation[i-1],\
            #     ker=ker, std=std, pad=pad, bn=False, dpc=dpc, bias = True)
            self.cnn.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            in_channels = out_channels

        for _ in range(1, n_extra_layers+1):
            self.exf +=cnn1d(nc, nc, activation[i-1],\
                ker=3, std=1, pad=1, bn=False, dpc=dpc, bias = bias)

        self.cnn = sqn(*self.cnn)
        self.exf = sqn(*self.exf)
        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 1),
            torch.nn.LeakyReLU(negative_slope=1.0, inplace=True)
        )

    def extraction(self, x):
        f = [self.exf[0](x)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            zlf = pll(self.cnn,x,self.gang)
            zfl = torch.reshape(X,(-1,1))
            zfl = pll(self.features_to_prob,zfl, self.gang)
            # torch.cuda.empty_cache()
        else:
            zlf = self.cnn(x)
        if not self.training:
            zlf=zlf.detach()
        return zlf

    def critic(self,X):
        X = self.forward(X)
        z =  torch.reshape(X,(-1,1))
        return pll(self.features_to_prob(z))