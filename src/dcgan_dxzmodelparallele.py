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

class DCGAN_DXZModelParallele(object):
    """docstring for DiscriminatorModelParallele"""
    def __init__(self, *arg, **kwargs):
        super(DCGAN_DXZModelParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DXZByGPU(ngpu, nc, nly,act,\
                 ker,std,pad,dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0, 
                 dpc=0.25, opt=None, *args, **kwargs):
        classname = 'DCGAN_DXZ_' + str(ngpu)+'GPU'
        #this following code is equivalent to calls the class it self. 
        """
        Here we define a methode, in which the purpose is to call and return a class by the name 
        This is made in the purpose of spliting the whole network in the GPUs allowed. There for 
        This suppose that one node of ngpu is present in the configuraton of the environement. 
        """  
        module_name = "dcgan_dxzmodelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module, classname)

        return class_(ngpu = ngpu, nc = nc, nly = nly, act=act, n_extra_layers = n_extra_layers,\
                     dpc = dpc, wf =wf, opt=opt, ker = ker, std = std, pad = pad, dil=dil,\
                     grp =grp, bn = bn)


class BasicDCGAN_DXZ(Module):
    """docstring for BasicDiscriminatorModelParallele"""
    def __init__(self):
        super(BasicDCGAN_DXZ, self).__init__()
        #Oridnal number of the GPUs
        self.dev0 = 0
        self.dev1 = 1
        self.dev2 = 2
        self.dev3 = 3

        #intialization of the cnns
        self.cnn1 = []
        self.cnn2 = []
        self.cnn3 = []
        self.cnn4 = []

        self.exf = []

        self.training = True

    def lout(self, nc, nly, increment):
        return nc if increment < nly else 1


class DCGAN_DXZ_1GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ"""
    def __init__(self, ngpu, nc, nly,act,\
                 ker=1,std=1,pad=0, dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0,
                 dpc=0.250, opt=None, *args, **kwargs):

        super(DCGAN_DXZ_1GPU, self).__init__()
        
        self.wf = wf
        #activation 
        activation = T.activation(act,nly)
        
        #initialisation of the input channel
        in_channels = nc

        for i in range(1, nly+1):
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1] if i<nly+1 else Sigmoid()
            self.cnn1 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        for i in range(1, n_extra_layers+1):
            self.exf +=cnn1d(nc,nc, activation[i-1],\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)

        self.cnn1 = sqn(*self.cnn1)
        self.exf = sqn(*self.exf)
        self.cnn1.to(self.dev0, dtype=torch.float32)
        self.exf.to(self.dev0, dtype=torch.float32)

    def extraction(self, x):
        f = [self.exf[0](x)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        # torch.cuda.empty_cache()
        if self.wf:
            f =  self.extraction(x)
        if not self.training:
            x=x.detach()
        if self.wf:
            return x,f
        else:
            return x

class DCGAN_DXZ_2GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ_2GPU"""
    def __init__(self, ngpu, nc, nly,act,\
                 ker=1,std=1,pad=0, dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0, 
                 dpc=0.250, opt=None, *args, **kwargs):
        super(DCGAN_DXZ_2GPU, self).__init__()
       
        self.wf = wf
        activation = T.activation(act,nly)

        in_channels = nc

        #part I in the GPU0
        for i in range(1, nly//2+1):
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            

        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        #This extra layer will be placed in the GPU0
        for i in range(1, n_extra_layers+1):
            self.exf +=cnn1d(nc,nc, activation[i-1],\
                ker=ker, std=std, pad=pad, opd=opd, bn=True, dpc=dpc)
    
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.exf =  sqn(*self.exf)
        self.exf.to(self.dev0, dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)

        # torche.cuda.empty_cache()
        if not self.training:
            x = x.detach()
        return x


class DCGAN_DXZ_3GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ_GPU"""
    def __init__(self, ngpu, nc, nly,act,\
                 ker=1,std=1,pad=0, dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0,\
                 dpc=0.250, opt=None, *arg, **kwargs):
        super(DCGAN_DXZ_3GPU, self).__init__()

        self.wf = wf
        #activation function
        activation = T.activation(act,nly)

        in_channels = nc
        
        #Part I GPU0
        for i in range(1, nly//3+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, bn=True, dpc=dpc)
            in_channels = out_channels    

        #Part II GPU1
        for i in range(nly//3+1, 2*nly//3+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels 

        #Part III GPU2
        for i in range(2*nly//3+1, nly+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn3 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        #This extra layer will be placed in the GPU0
        for i in range(1, n_extra_layers+1):
            self.exf +=cnn1d(nc,nc, activation[i-1],\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2, dtype=torch.float32)

        self.exf =  sqn(*self.exf)
        self.exf.to(self.dev0, dtype=torch.float32)

    def extraction(self, x):
            x.to(self.dev0,  dtype=torch.float32)
            f = [self.exf[0](x)]
            for l in range(1,len(self.exf)):
                f.append(self.exf[l](f[l-1]))
            return f 

    def forward(self,x):
            x = x.to(self.dev0,dtype=torch.float32)
            x = self.cnn1(x)
            x = x.to(self.dev1,dtype=torch.float32)
            x = self.cnn2(x)
            x = x.to(self.dev2,dtype=torch.float32)
            x = self.cnn3(x)
            # torch.cuda.empty_cache()
            if self.wf:
                f =  self.extraction(x)
            if not self.training:
                x=x.detach()
            if self.wf:
                return x,f
            else:
                return x

class DCGAN_DXZ_4GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ_4GPU"""
    def __init__(self, ngpu, nc, nly,act,\
                 ker=1,std=1,pad=0, dil=1, grp=1, bn=True,wf=False,\
                 n_extra_layers=0,\
                 dpc=0.250, opt=None, *args, **kwargs):
        super(DCGAN_DXZ_4GPU, self).__init__()
        
        self.wf = wf

        #activation functions 
        activation = T.activation(act,nly)

        in_channels = nc
        #Part I in GPU0
        for i in range(1, nly//4+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels

        #Part II in GPU1
        for i in range(nly//3+1, 2*nly//4+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels

        #Part III in GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn3 += cnn1d(in_channels, out_channels, act,\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)
            in_channels = out_channels

        #Part IV in GPU3
        for i in range(3*nly//4+1, nly+1):
            out_channels = self.lout(nc, nly, i)
            act = activation[i-1]
            self.cnn4 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
                    bn=False,dpc=0.0,wn=False)
            in_channels = out_channels

         
        #This extra layer will be placed in the GPU0
        for i in range(1, n_extra_layers+1):
            self.exf +=cnn1d(nc,nc, activation[i-1],\
                ker=ker, std=std, pad=pad, bn=True, dpc=dpc)

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2, dtype=torch.float32)

        self.cnn4 = sqn(*self.cnn4)
        self.cnn4.to(self.dev3, dtype=torch.float32)
        
        self.exf =  sqn(*self.exf)
        self.exf.to(self.dev0, dtype=torch.float32)

    def extraction(self, x):
        x.to(self.dev0,  dtype=torch.float32)
        f = [self.exf[0](x)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

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
        if self.wf:
            f =  self.extraction(x)
        if not self.training:
            x = x.detach()
        if self.wf:
            return x,f
        else:
            return x