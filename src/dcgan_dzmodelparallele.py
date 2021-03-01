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
    def getDCGAN_DzByGPU(ngpu, nz, nly, act, channel, ncl=512, fpd=0, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False,limit = 256, bias =  False):
        #we assign the code name
        classname = 'DCGAN_Dz_' + str(ngpu)+'GPU'

        module_name = "dcgan_dzmodelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module,classname)

        return class_(ngpu = ngpu, nz = nz, nly = nly, act=act, ncl = ncl,channel = channel,\
                    n_extra_layers = n_extra_layers, dpc =dpc,\
                    ker=ker, std=std, pad = pad, dil=dil, grp=grp,\
                    bn =bn, wf = wf, limit = limit, bias = bias)


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
            self.wf = True
            self.prc = []
            self.exf = []

        def lout(self, nz, nly, ncl, increment, limit):
            #this is the logic of in_channels and out_channels
            # val = ncl if increment!=nly-1 else nz
            val = limit
            return val if val<=limit else limit 

        def extraction(self,X):
            pass

class  DCGAN_Dz_1GPU(BasicDCGAN_Dz):
    """docstring for  DCGAN_Dz_1GPU"""
    def __init__(self, ngpu, nz, nly,act, channel,  ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False,limit = 256, bias = False):
        super(DCGAN_Dz_1GPU, self).__init__()

        #activation functions
        activation = T.activation(act, nly)

        self.wf = wf

        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2,nly+1):
            
            act = activation[i-1]
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            # self.cnn1 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
            #         bn=_bn,dpc=dpc,wn=False) 

        for _ in range(0,n_extra_layers):
            self.cnn1.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

        self.exf = self.cnn1[:-1]
        self.cnn1.append(Dpout(dpc = dpc))
        self.cnn1.append(activation[-1])
        self.cnn1 = self.prc + self.cnn1
        
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype = torch.float32)

        self.prc = sqn(*self.prc)
        self.prc.to(self.dev0, dtype = torch.float32)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self, x):
        z = x.to(self.dev0, dtype=torch.float32)
        z = self.cnn1(x)
        torch.cuda.empty_cache()
        if self.wf:
            f = self.extraction(x)
        if not self.training:
            z=z.detach()
        if self.wf:
            return z,f
        else:
            return z

class DCGAN_Dz_2GPU(BasicDCGAN_Dz):
    """docstring for DCGAN_Dz_2GPU"""
    def __init__(self, ngpu, nz, nly, act, channel,ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False,limit = 256, bias = False):
        super(DCGAN_Dz_2GPU, self).__init__()
        
        #activation functions
        activation = T.activation(act, nly)

        self.wf = wf

        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        #Part I is in GPU0
        for i in range(2,nly//2+1):
            
            act = activation[i-1]
            # _bn =  False if i == 1 else bn
            # self.cnn1 += cnn1d(in_channels, out_channels, act,\
            #     ker=ker, std=std, pad=pad,bn=_bn, dpc=dpc)
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        #Part II is in GPU1
        for i in range(nly//2+1,nly+1):
            
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
            #         bn=bn,dpc=dpc,wn=False)
            self.cnn2.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for _ in range(0,n_extra_layers):
            self.cnn2.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

        self.exf = self.cnn2[:-1]

        self.cnn2.append(Dpout(dpc = dpc))
        self.cnn2.append(activation[-1])

        self.cnn1 = self.prc + self.cnn1

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)

        self.prc = sqn(*self.prc)
        self.prc.to(self.dev0, dtype = torch.float32)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,x):
        z = x.to(self.dev0,dtype = torch.float32)
        z = self.cnn1(z)

        z = z.to(self.dev1,dtype = torch.float32)
        z = self.cnn2(z)

        if self.wf:
            f = self.extraction(x)
        if not self.training:
            z=z.detach()
        if self.wf:
            return z,f
        else:
            return z

class DCGAN_Dz_3GPU(BasicDCGAN_Dz):
    """docstring for DCGAN_Dz_3GPU"""
    def __init__(self, ngpu, nz, nly, act, channel, ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, limit = 256, bias = False):
        super(DCGAN_Dz_3GPU, self).__init__()

        #activation 
        activation = T.activation(act, nly)

        self.wf = wf

        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2, nly//3+1):
            
            act = activation[i-1]
            # _bn =  False if i == 1 else bn
            # self.cnn1 += cnn1d(in_channels, out_channels, act,\
            #     ker=ker, std=std,bn=_bn, dpc=dpc)
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for i in range(nly//3+1, 2*nly//3+1):
            
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels, out_channels, act,\
            #     ker=ker, std=std, pad=pad,bn=bn, dpc=dpc)
            self.cnn2.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for i in range(2*nly//3+1, nly+1):
            
            act = activation[i-1]
            # self.cnn3 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
            #         bn=bn,dpc=dpc,wn=False)
            self.cnn3.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for _ in range(0,n_extra_layers):
            self.cnn3.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

        self.exf = self.cnn3[:-1]
        self.cnn3.append(Dpout(dpc = dpc))
        self.cnn3.append(activation[-1])

        self.cnn1 = self.prc + self.cnn1

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)

        self.prc = sqn(*self.prc)
        self.prc.to(self.dev0, dtype = torch.float32)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
        
    def forward(self, x):
        z = x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(z)
        z = z.to(self.dev1,dtype=torch.float32)
        z = self.cnn2(z)
        z = z.to(self.dev2,dtype=torch.float32)
        z = self.cnn3(z)

        if self.wf:
            f = self.extraction(x)
        if not self.training:
            z=z.detach()
        if self.wf:
            return z,f
        else:
            return z

class DCGAN_Dz_4GPU(BasicDCGAN_Dz):
    """docstring for DCGAN_Dz_4GPU"""
    def __init__(self, ngpu, nz, nly, act,channel, ncl=512, fpd=1, n_extra_layers=1, dpc=0.25,
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False,limit = 256, bias = False):
        super(DCGAN_Dz_4GPU, self).__init__()
        
        #activation 
        activation = T.activation(act, nly)

        self.wf = wf
        
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2, nly//4+1):
            
            act = activation[i-1]
            # _bn =  False if i == 1 else bn
            # self.cnn1 += cnn1d(in_channels, out_channels, act,\
            #     ker=ker, std=std, pad=pad, bn=_bn, dpc=dpc)
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = out_channels,
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for i in range(nly//4+1, 2*nly//4+1):
            
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels, out_channels, act,\
            #     ker=ker, std=std, pad=pad, bn=bn, dpc=dpc)
            self.cnn2.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for i in range(2*nly//4+1, 3*nly//4+1):
            
            act = activation[i-1]
            # self.cnn3 += cnn1d(in_channels, out_channels, act,\
            #     ker=ker, std=std, pad=pad, bn=bn, dpc=dpc)
            self.cnn3.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for i in range(3*nly//4+1, nly+1):
            
            act = activation[i-1]
            # self.cnn4 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
            #         bn=bn,dpc=dpc,wn=False)
            self.cnn4.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], bias = bias, act = act, dpc = dpc, bn=bn))
            

        for _ in range(0,n_extra_layers):
            self.cnn1.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                ks = 3, stride = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = act))

        self.exf = self.cnn4[:-1]
        self.cnn4.append(Dpout(dpc = dpc))
        self.cnn4.append(activation[-1])

        self.cnn1 = self.prc + self.cnn1

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)
        self.cnn4 = sqn(*self.cnn4)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)
        self.cnn4.to(self.dev3,dtype=torch.float32)

        self.prc = sqn(*self.prc)
        self.prc.to(self.dev0, dtype = torch.float32)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self, x):
        z = x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(z)
        z = z.to(self.dev1,dtype=torch.float32)
        z = self.cnn2(z)
        z = z.to(self.dev2,dtype=torch.float32)
        z = self.cnn3(z)
        z = z.to(self.dev3,dtype=torch.float32)
        z = self.cnn4(z)
        # torch.cuda.empty_cache()
        if self.wf:
            f = self.extraction(x)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z
