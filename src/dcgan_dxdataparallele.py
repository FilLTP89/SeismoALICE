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
    def getDCGAN_DxDataParallele(name, ngpu, nc, ncl, ndf, nly,act,fpd=0, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0,isize=256):
        if name is not None:
            classname = 'DCGAN_Dx_'+ name
            #preparation for other DataParallele Class
            try:
                module_name = "dcgan_dxdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)
                return class_(ngpu=ngpu, isize=isize, nc=nc, ncl=ncl, ndf=ndf, fpd=fpd, act=act, nly=nly,\
                 ker=ker,std=std,pad=pad, dil=dil,grp=grp,bn=bn,wf=wf, dpc=dpc, limit = limit,\
                 n_extra_layers=n_extra_layers)
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return DCGAN_Dx(ngpu = ngpu, isize = isize, nc = nc, ncl = ncl, ndf = ndf, fpd = fpd,\
                        nly = nly, ker=ker ,std=std, pad=pad, dil=dil, grp=grp, bn=bn, wf = wf, dpc=dpc,\
                        n_extra_layers = n_extra_layers,limit = limit)

class BasicDCGAN_DxDataParallele(Module):
    """docstring for BasicDCGAN_DxDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DxDataParallele, self).__init__()
        self.training = True

    def lout(self,nz, nly, increment, limit):
        #Here we specify the logic of the  in_channels/out_channels
        n = nz*2**(increment)
        #we force the last of the out_channels to not be greater than 512
        val = n if (n<limit or increment<nly) else limit
        return val if val<=limit else limit
        

class   DCGAN_Dx(BasicDCGAN_DxDataParallele):
    """docstring for    DCGAN_Dx"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, fpd=1, isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0):

        super(DCGAN_Dx, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.cnn  = []

        #activation code
        activation = T.activation(act,nly)

        in_channels =  nc
        for i in range(1, nly+1):
            out_channels =  self.lout(ndf, nly, i, limit)
            _bn = False if i == 1 else bn
            _dpc = 0.0 if i == nly else dpc
            act = activation[i-1]
            # self.cnn += cnn1d(in_channels, out_channels,act,\
            #     ker=ker, std=std, pad=pad, dil=dil, bn=_bn, dpc=_dpc )
            _bn = bn if i == 1 else True
            self.cnn.append(ConvBlock(ni = in_channels, no = out_channels,
                ks = ker, stride = std, pad = pad, dil = dil, bias = False,\
                bn = _bn, dpc = dpc, act = act))
            in_channels = out_channels

        for i in range(0,n_extra_layers):
            self.cnn.append(ConvBlock(ni = in_channels,no=in_channels,\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

        self.cnn = sqn(*self.cnn)


    def foward(self,x):
        if x.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,x,self.gang)
            # torch.cuda.empty_cache()
        else:
            zlf   = self.cnn(x)
        if not self.training:
            zlf=zlf.detach()
        return zlf
        
        