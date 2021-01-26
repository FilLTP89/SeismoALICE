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
import importlib


class DCGAN_DzModelParallele(object):
    """docstring for DCGAN_Dz"""
    def __init__(self, *args, **kwargs):
        super(DCGAN_DzModelParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DzByGPU(ngpu, nz, nly, act, ncl=512, fpd=0, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False):
        #we assign the code name
        classname = 'DCGAN_Dz_' + str(ngpu)+'GPU'

        module_name = "dcgan_dzmodelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module,classname)

        return class_(ngpu = ngpu, nz = nz, nly = nly, act=act, ncl = ncl, n_extra_layers = n_extra_layers, dpc =dpc,\
            ker=ker, std=std, pad = pad, dil=dil, grp=grp, bn =bn, wf = wf)


class BasicDCGAN_Dz(Module):
        """docstring for BasicDCGAN_Dz"""
        def __init__(self):
            super(BasicDCGAN_Dz, self).__init__()
            #ordinal value of the GPUs
            self.dev0 = 0
            self.dev1 = 1
            self.dev2 = 2
            self.dev3 = 3

            #initial
            self.cnn1 = []
            self.cnn2 = []
            self.cnn3 = []
            self.cnn4 = []

            #training test
            self.training =  True

        def lout(self, nz, nly,ncl, increment):
            #this is the logic of in_channels and out_channels
            return ncl if increment!=nly-1 else nz


class  DCGAN_Dz_1GPU(BasicDCGAN_Dz):
    """docstring for  DCGAN_Dz_1GPU"""
    def __init__(self, ngpu, nz, nly,act,  ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False):
        super(DCGAN_Dz_1GPU, self).__init__()

        #activation functions
        activation = T.activation(act, nly)

        #initialization of the input channel
        in_channels  = nz

        for i in range(1,nly+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype = torch.float32)

    def forward(self, x):
        
        x = x.to(self.dev0, dtype=torch.float32)
        x = self.cnn1(x)
        torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x

class DCGAN_Dz_2GPU(BasicDCGAN_Dz):
    """docstring for DCGAN_Dz_2GPU"""
    def __init__(self, ngpu, nz, nly, act, ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False):
        super(DCGAN_Dz_2GPU, self).__init__()
        
        #activation functions
        activation = T.activation(act, nly)

        in_channels = nz

        #Part I is in GPU0
        for i in range(1,nly//2+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad,bn=True, dpc=dpc)
            in_channels = out_channels

        #Part II is in GPU1
        for i in range(nly//2+1,nly+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)

    def forward(self,x):

        x = x.to(self.dev0,dtype = torch.float32)
        x = self.cnn1(x)

        x = x.to(self.dev1,dtype = torch.float32)
        x = self.cnn2(x)
        torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x

class DCGAN_Dz_3GPU(BasicDCGAN_Dz):
    """docstring for DCGAN_Dz_3GPU"""
    def __init__(self, ngpu, nz, nly, act, ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False):
        super(DCGAN_Dz_3GPU, self).__init__()

        #activation 
        activation = T.activation(act, nly)

        in_channels = nz

        for i in range(1, nly//3+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std,n=True, dpc=dpc)
            in_channels = out_channels

        for i in range(nly//3+1, 2*nly//3+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad,bn=True, dpc=dpc)
            in_channels = out_channels

        for i in range(2*nly//3+1, nly+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn3 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)
        
    def forward(self, x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)
        x = x.to(self.dev2,dtype=torch.float32)
        x = self.cnn3(x)
        torch.cuda.empty_cache()
        if not self.training:
            x = x.detach()
        return x

class DCGAN_Dz_4GPU(BasicDCGAN_Dz):
    """docstring for DCGAN_Dz_4GPU"""
    def __init__(self, ngpu, nz, nly, act, ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False):
        super(DCGAN_Dz_4GPU, self).__init__()
        
        #activation 
        activation = T.activation(act, nly)
        
        in_channels = nz

        for i in range(1, nly//4+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels

        for i in range(nly//4+1, 2*nly//4+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels

        for i in range(2*nly//4+1, 3*nly//4+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn3 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels

        for i in range(3*nly//4+1, nly+1):
            out_channels = self.lout(nz, nly, ncl, i)
            act = activation[i-1]
            self.cnn4 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=False,dpc=0.0,wn=False)
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)
        self.cnn4 = sqn(*self.cnn4)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)
        self.cnn4.to(self.dev3,dtype=torch.float32)

    def forward(self, x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)
        x = x.to(self.dev2,dtype=torch.float32)
        x = self.cnn3(x)
        x = x.to(self.dev3,dtype=torch.float32)
        x = self.cnn4(x)
        # torch.cuda.empty_cache()
        if not self.training:
            x = x.detach()
        return x
