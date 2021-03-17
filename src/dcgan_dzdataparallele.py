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

class DCGAN_DzDataParallele(object):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,*args, **kwargs):
        super(DCGAN_DzDataParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DzDataParallele(name,ngpu, nc,nz, act, ncl, ndf, path, nly, channel, fpd=1,\
                 ker=2,std=2,pad=0, dil=0,grp=0,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0, limit = 256, bias = False):
        if name is not None:
            classname = 'DCGAN_Dz_' + name
            #prepraring calling class by name if the name exist
            try:
                module_name = "dcgan_dzdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)

                return class_(ngpu = ngpu, nc=nc, nz =nz, ncl=ncl, ndf=ndf, act=act,fpd=fpd, nly = nly,channel = channel,\
                 ker=ker,std=std,pad=pad, path=path, dil=dil,grp=grp,bn=bn,wf=wf, dpc=dpc, limit = limit,bias = bias,
                 n_extra_layers=n_extra_layers)
            except Exception as e:
                raise e
                print("The class ", classname, " does not exit")
        else:
            return DCGAN_Dz(ngpu, nc=nc, ncl=ncl, nz=nz, ndf = ndf, act=act,fpd=fpd, nly = nly, channel = channel,\
                 ker=ker,std=std,pad=pad, dil=dil, path=path, grp=grp, bn=bn,wf=wf, dpc=dpc, limit = limit,bias = bias,\
                 n_extra_layers=n_extra_layers)


class BasicDCGAN_DzDataParallele(Module):
    """docstring for BasicDCGAN_DzDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DzDataParallele, self).__init__()
        self.training =  True
        self.prc = []
        self.exf = []

    def lout(self, nz, nly,ncl, increment, limit):
        #this is the logic of in_channels and out_channels
        val =  ncl if increment!=nly-1 else nz
        return val if val <=limit else limit

class DCGAN_Dz(BasicDCGAN_DzDataParallele):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,ngpu, nc, ndf, nz, nly, act, channel, fpd=0,ncl = 512,limit = 512,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250, bias = False,
                 n_extra_layers=0, path=''):

        super(DCGAN_Dz, self).__init__()

        device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        #activation functions
        activation = T.activation(act, nly)
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf  = wf
        self.net = []
        layers   = []

        # pdb.set_trace()
        if path:
            self.cnn1 = T.load_net(path)
            # Freeze model weights
            for param in self.cnn1.parameters():
                param.requires_grad = False

        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2,nly+1):
            
            act = activation[i-1]
            self.net.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            # self.cnn1 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
            #         bn=_bn,dpc=dpc,wn=False)
            layers.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                    ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                    dpc = dpc, act = act))

        for k in range(0,n_extra_layers):
            self.net.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

        # pdb.set_trace()
        self.exf = layers[:-1]
        self.exf =sqn(*self.exf).to(device)

        self.net.append(Dpout(dpc = dpc))
        self.net.append(activation[-1])
        self.net = self.prc + self.net
        self.net =sqn(*self.net)
        
        # pdb.set_trace()

        if path:
            self.cnn1.cnn1[2] = copy.deepcopy(self.net[0:-2])
        else:
            self.cnn1 = copy.deepcopy(self.net)
        del self.net

        self.cnn1.to(device)

        self.prc = sqn(*self.prc)
        self.prc.to(device)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.cnn1,x,self.gang)
            z = T._forward(X, self.cnn1, self.gang)
            if self.wf:
                f = pll(self.extraction,X,self.gang)
                # f = self.extraction(X)
        else:
            z = self.cnn1(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        
        if self.wf:
            return z,f
        else:
            return z