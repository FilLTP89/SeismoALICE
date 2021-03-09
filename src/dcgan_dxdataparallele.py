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


class DCGAN_DxDataParallele(object):
    """docstring for DCGAN_DxDataParallele"""
    def __init__(self, *arg, **kwargs):
        super(DCGAN_DxDataParallele, self).__init__()
        pass
    @staticmethod
    def getDCGAN_DxDataParallele(name, ngpu, nc, ncl, ndf, nly,act,channel,fpd=0, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0,isize=256):
        if name is not None:
            classname = 'DCGAN_Dx_'+ name
            #preparation for other DataParallele Class
            try:
                module_name = "dcgan_dxdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)
                return class_(ngpu=ngpu, isize=isize, nc=nc, ncl=ncl, ndf=ndf,  channel = channel, fpd=fpd, act=act, nly=nly,\
                 ker=ker,std=std,pad=pad, dil=dil,grp=grp,bn=bn,wf=wf, dpc=dpc, limit = limit,\
                 n_extra_layers=n_extra_layers)
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return DCGAN_Dx(ngpu = ngpu, isize = isize, nc = nc, ncl = ncl, ndf = ndf, fpd = fpd,channel = channel,\
                        nly = nly, ker=ker ,std=std, pad=pad, dil=dil, grp=grp, bn=bn, wf = wf, dpc=dpc,\
                        n_extra_layers = n_extra_layers,limit = limit)

class BasicDCGAN_DxDataParallele(Module):
    """docstring for BasicDCGAN_DxDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DxDataParallele, self).__init__()
        self.training = True
        self.wf       = True
        self.prc      = []
        self.exf      = []
        self.extra    = []
        self.final    = []

    def lout(self,nz, nly, increment, limit):
        #Here we specify the logic of the  in_channels/out_channels
        n = nz*2**(increment)
        #we force the last of the out_channels to not be greater than 512
        val = n if (n<limit or increment<nly) else limit
        return val if val<=limit else limit
        

class   DCGAN_Dx(BasicDCGAN_DxDataParallele):
    """docstring for    DCGAN_Dx"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel, fpd=1, isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0):

        super(DCGAN_Dx, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.cnn  = []

        #activation code
        activation = T.activation(act,nly)

        #extraction features 
        self.wf = wf

        #building network
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2, nly+1):
            
            _bn = False if i == 1 else bn
            _dpc = 0.0 if i == nly else dpc
            act = activation[i-1]
            # self.cnn += cnn1d(in_channels, out_channels,act,\
            #     ker=ker, std=std, pad=pad, dil=dil, bn=_bn, dpc=_dpc )
            _bn = bn if i == 1 else True
            self.cnn.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1],\
                bias = False,\
                bn = _bn, dpc = dpc, act = act))

        for _ in range(0,n_extra_layers):
            self.extra+=[Conv1d(in_channels = channel[i],out_channels=channel[i],\
                kernel_size = 3, stride = 1, padding=1, bias=False)]

        self.final+=[Conv1d(channel[i], channel[i], 3, padding=1, bias=False)]
        self.final+=[BatchNorm1d(channel[i])]
        self.final+=[Dpout(dpc=dpc)]
        self.final+=[activation[-1]]

        self.exf = self.cnn
        self.cnn = self.prc + self.cnn + self.extra + self.final

        self.cnn = sqn(*self.cnn).to(device)
        self.prc = sqn(*self.prc).to(device)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            # z = pll(self.cnn,X,self.gang)
            z = T._forward(X, self.cnn, self.gang)
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.cnn(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z