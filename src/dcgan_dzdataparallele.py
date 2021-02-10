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

class DCGAN_DzDataParallele(object):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,*args, **kwargs):
        super(DCGAN_DzDataParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DzDataParallele(name,ngpu, nc,nz, act, ncl, ndf, nly, fpd=1,\
                 ker=2,std=2,pad=0, dil=0,grp=0,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0, limit = 256, bias = False):
        if name is not None:
            classname = 'DCGAN_Dz_' + name
            #prepraring calling class by name if the name exist
            try:
                module_name = "dcgan_dzdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)

                return class_(ngpu = ngpu, nc=nc, nz =nz, ncl=ncl, ndf=ndf, act=act,fpd=fpd, nly = nly,\
                 ker=ker,std=std,pad=pad, dil=dil,grp=grp,bn=bn,wf=wf, dpc=dpc, limit = limit,bias = bias,
                 n_extra_layers=n_extra_layers)
            except Exception as e:
                raise e
                print("The class ", classname, " does not exit")
        else:
            return DCGAN_Dz(ngpu, nc=nc, ncl=ncl, nz=nz, ndf = ndf, act=act,fpd=fpd, nly = nly,\
                 ker=ker,std=std,pad=pad, dil=dil, grp=grp, bn=bn,wf=wf, dpc=dpc, limit = limit,bias = bias,\
                 n_extra_layers=n_extra_layers)


class BasicDCGAN_DzDataParallele(Module):
    """docstring for BasicDCGAN_DzDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DzDataParallele, self).__init__()
        self.training =  False

    def lout(self, nz, nly,ncl, increment, limit):
        #this is the logic of in_channels and out_channels
        val =  ncl if increment!=nly-1 else nz
        return val if val <=limit else limit

class DCGAN_Dz(BasicDCGAN_DzDataParallele):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,ngpu, nc, ndf, nz, nly, act, fpd=0,ncl = 512,limit = 512,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250, bias = False,
                 n_extra_layers=0):

        super(DCGAN_Dz, self).__init__()

        #activation functions
        activation = T.activation(act,nly)

        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.cnn  = []

        in_channels =  nz

        for i in range(1,nly+1):
            out_channels = self.lout(nz, nly, ncl, i, limit)
            # _bn =  False if i == 1 else bn
            # self.cnn += cnn1d(in_channels, out_channels, activation[i-1],\
            #     ker=ker, std=std, pad=pad, bn=bn, dpc=dpc)
            self.cnn1.append(ConvBlock(ni = in_channels, no = out_channels,
                ks = ker, stride = std, pad = pad, bias = bias, act = act, dpc = dpc, bn=bn))
            in_channels = out_channels

        self.cnn = sqn(*self.cnn)

        for i in range(0,n_extra_layers):
            self.cnn1.append(ConvBlock(ni = in_channels,no=in_channels,\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,x,self.gang)
            torch.cuda.empty_cache()
        else:
            zlf   = self.cnn(x)
        if not self.training:
            zlf=zlf.detach()
        return zlf  