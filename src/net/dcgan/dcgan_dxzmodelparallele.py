from torch.nn.modules import activation

u'''AE design'''
from torch.nn.modules import activation
u'''AE design'''
u'''Required modules'''
import warnings
# import GPUtil
warnings.filterwarnings("ignore")
from common.common_nn import *
import torch
import pdb
from torch import device as tdev
import importlib
import copy

class DCGAN_DXZModelParallele(object):
    """docstring for DiscriminatorModelParallele"""
    def __init__(self, *arg, **kwargs):
        super(DCGAN_DXZModelParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DXZByGPU(ngpu, nc, nly,act, channel,\
                 ker,std,pad,dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0, limit = 256, path='',
                 dpc=0.25, bias = True, opt=None, *args, **kwargs):
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

        return class_(ngpu = ngpu, nc = nc, nly = nly, act=act, channel = channel,\
                     n_extra_layers = n_extra_layers, path=path,\
                     dpc = dpc, wf =wf, opt=opt, ker = ker, std = std, pad = pad, dil=dil,\
                     grp =grp, bn = bn, limit = limit, bias = bias)


class BasicDCGAN_DXZ(torch.nn.Module):
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
        self.wf = True
        self.splits = 1

        self.training = True

    def lout(self, nc, nly, increment, limit):
        val =  nc if increment < nly else 1
        return val if val<=limit else limit

    def critic(self,x):
        pass

    def extraction(self,x):
        pass

class DCGAN_DXZ_1GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ"""
    def __init__(self, ngpu, nc, nly,act, channel,\
                 ker=1,std=1,pad=0, dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0,limit = 1024,path='',\
                 dpc=0.250, bias = True, opt=None, *args, **kwargs):

        super(DCGAN_DXZ_1GPU, self).__init__()
        
        self.wf = wf
        self.net = []
        #activation 
        activation = T.activation(act,nly)
        # pdb.set_trace()
        if path:
            self.cnn1 = T.load_net(path)
            # Freeze model weights
            for param in self.cnn1.parameters():
                param.requires_grad = False
        #initialisation of the input channel
        
        for i in range(1, nly+1):
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            
            act = activation[i-1]
            self.net.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ker = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            # self.cnn1 += cnn1d(in_channels,nc, act, ker=ker,std=std,pad=pad,\
            #         bn=False,bias=True, dpc=dpc,wn=False)

        for _ in range(1, n_extra_layers+1):
            self.net +=cnn1d(nc,nc, activation[i-1],\
                ker=3, std=1, pad=1, bn=False, bias =bias, dpc=dpc, wn = False)
        # any changes made to a copy of object do not reflect in the original object
        self.exf = copy.deepcopy(self.net[:-1])

        self.net = sqn(*self.net)
        # pdb.set_trace()
        if path:
            self.cnn1.cnn1[-1] = copy.deepcopy(self.net[:])
        else: 
            self.cnn1 = copy.deepcopy(self.net)
        del self.net

        self.exf = sqn(*self.exf)

        self.cnn1.to(self.dev0, dtype=torch.float32)
        self.exf.to(self.dev0, dtype=torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(nc, 1),
            #activation, leakyReLU if WGAN and sigmoid if GAN
            activation[-1]
        ).to(device).to(self.dev0, dtype=torch.float32)

    def extraction(self, x):
        f = [ T._forward_1G(x,self.exf[0])]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,x):
        # pdb.set_trace()
        z = x.to(self.dev0,dtype=torch.float32)
        # z = self.cnn1(z)
        z =  T._forward_1G(z,self.cnn1)
        z =  torch.reshape(z,(-1,1))
        z =  T._forward(z,self.features_to_prob) 
        torch.cuda.empty_cache()
        if self.wf:
            f =  self.extraction(x)
        if not self.training:
            z = z.detach()

        if self.wf:
            return z,f
        else:
            return z

    def critic(self,X):
        X = self.forward(X)
        z =  torch.reshape(X,(-1,1))
        return self.features_to_prob(z)


class DCGAN_DXZ_2GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ_2GPU"""
    def __init__(self, ngpu, nc, nly,act, channel,\
                 ker=1,std=1,pad=0, dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0, limit = 1024,path='',\
                 dpc=0.250, bias = True, opt=None, *args, **kwargs):
        super(DCGAN_DXZ_2GPU, self).__init__()
       
        self.wf = wf
        activation = T.activation(act,nly)

        #part I in the GPU0
        for i in range(1, nly//2+1):
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            act = activation[i-1]
            # self.cnn1 += cnn1d(in_channels, nc, act,\
            #     ker=ker, std=std, pad=pad, bn=False, bias=True, dpc=dpc)
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            
        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels,nc, act, ker=ker,std=std,pad=pad,\
            #         bn=False,dpc=dpc,wn=False, bias=True)
            self.cnn2.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            
        #This extra layer will be placed in the GPU0
        for _ in range(1, n_extra_layers+1):
            self.cnn2 +=cnn1d(nc,nc, activation[i-1],\
                ker=3, std=1, pad=1, bn=False, dpc=dpc, bias=bias)
        # pdb.set_trace()
        self.exf = copy.deepcopy(self.cnn2[:-1])

        self.cnn1 = sqn(*self.cnn1).to(self.dev0,dtype=torch.float32)
        self.cnn2 = sqn(*self.cnn2).to(self.dev1,dtype=torch.float32)

        #self.cnn2.to(self.dev1,dtype=torch.float32)
        #self.cnn1.to(self.dev0,dtype=torch.float32)
        
        self.exf =  sqn(*self.exf)
        self.exf.to(self.dev0, dtype=torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(nc, 1),
            torch.nn.LeakyReLU(negative_slope=1.0, inplace=True)
        ).to(self.dev1, dtype=torch.float32)

    def extraction(self, x):
        f = [ T._forward_1G(x,self.exf[0])]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,x):
        # pdb.set_trace()
        
        x = x.to(self.dev0,dtype=torch.float32)
        # z = self.cnn1(x)
        # z = z.to(self.dev1,dtype=torch.float32)
        # z = self.cnn2(z)
        z = T._forward_2G(x, self.cnn1, self.cnn2)
        if self.wf:
            f =  self.extraction(x)
        if not self.training:
            z = z.detach()

        if self.wf:
            return z,f
        else:
            return z

    def critic(self,X):
        X = self.forward(X)
        z =  torch.reshape(X,(-1,1))
        return self.features_to_prob(z)


class DCGAN_DXZ_3GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ_GPU"""
    def __init__(self, ngpu, nc, nly,act, channel,\
                 ker=1,std=1,pad=0, dil=1,grp=1,bn=True,wf=False,\
                 n_extra_layers=0,limit = 1024,path='',\
                 dpc=0.250,bias = True, opt=None, *arg, **kwargs):
        super(DCGAN_DXZ_3GPU, self).__init__()

        self.wf = wf
        #activation function
        activation = T.activation(act,nly)

        
        
        #Part I GPU0
        for i in range(1, nly//3+1):
            
            act = activation[i-1]
            # self.cnn1 += cnn1d(in_channels, nc, act,\
            #     ker=ker, std=std, bn=False, dpc=dpc, bias=True)
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
                

        #Part II GPU1
        for i in range(nly//3+1, 2*nly//3+1):
            
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels, nc, act,\
            #     ker=ker, std=std, pad=pad, bn=False, dpc=dpc, bias=True)
            self.cnn2.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
             

        #Part III GPU2
        for i in range(2*nly//3+1, nly+1):
            
            act = activation[i-1]
            # self.cnn3 += cnn1d(in_channels,nc, act, ker=ker,std=std,pad=pad,\
            #         bn=False,dpc=dpc,wn=False, bias=True)
            self.cnn3.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            

        #This extra layer will be placed in the GPU0
        for i in range(1, n_extra_layers+1):
            self.cnn3 +=cnn1d(nc,nc, activation[i-1],\
                ker=3, std=1, pad=1, bn=False, dpc=dpc, bias=bias)

        self.exf = copy.deepcopy(self.cnn3[:-1])

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2, dtype=torch.float32)

        self.exf =  sqn(*self.exf)
        self.exf.to(self.dev0, dtype=torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(nc, 1),
            torch.nn.LeakyReLU(negative_slope=1.0, inplace=True)
        ).to(self.dev2, dtype=torch.float32)

    def extraction(self, x):
        x.to(self.dev0,  dtype=torch.float32)
        f = [self.exf[0](x)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f 

    def forward(self,x):
        z = x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(z)
        z = z.to(self.dev1,dtype=torch.float32)
        z = self.cnn2(z)
        z = z.to(self.dev2,dtype=torch.float32)
        z = self.cnn3(z)
        
        if self.wf:
            f =  self.extraction(x)
        if not self.training:
            z = z.detach()

        if self.wf:
            return z,f
        else:
            return z

    def critic(self,X):
        X = self.forward(X)
        z =  torch.reshape(X,(-1,1))
        return self.features_to_prob(z)

class DCGAN_DXZ_4GPU(BasicDCGAN_DXZ):
    """docstring for DCGAN_DXZ_4GPU"""
    def __init__(self, ngpu, nc, nly,act,\
                 ker=1,std=1,pad=0, dil=1, grp=1, bn=True,wf=False,\
                 n_extra_layers=0,limit = 1024,path='',\
                 dpc=0.250,bias = True, opt=None, *args, **kwargs):
        super(DCGAN_DXZ_4GPU, self).__init__()
        
        self.wf = wf

        #activation functions 
        activation = T.activation(act,nly)

        
        #Part I in GPU0
        for i in range(1, nly//4+1):
            
            act = activation[i-1]
            # self.cnn1 += cnn1d(in_channels, nc, act,\
            #     ker=ker, std=std, pad=pad, bn=False, dpc=dpc, bias = True)
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            

        #Part II in GPU1
        for i in range(nly//3+1, 2*nly//4+1):
            
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels, nc, act,\
            #     ker=ker, std=std, pad=pad, bn=False, dpc=dpc, bias=True)
            self.cnn2.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            

        #Part III in GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            
            act = activation[i-1]
            # self.cnn3 += cnn1d(in_channels, nc, act,\
            #     ker=ker, std=std, pad=pad, bn=False, dpc=dpc, bias =True)
            self.cnn3.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            

        #Part IV in GPU3
        for i in range(3*nly//4+1, nly+1):
            
            act = activation[i-1]
            # self.cnn4 += cnn1d(in_channels,nc, act, ker=ker,std=std,pad=pad,\
            #         bn=False,dpc=dpc,wn=False, bias=True)
            self.cnn4.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = bn, dpc = dpc, act = act))
            
        #This extra layer will be placed in the GPU0
        for i in range(1, n_extra_layers+1):
            self.cnn4 +=cnn1d(nc,nc, activation[i-1],\
                ker=3, std=1, pad=1, bn=True, dpc=dpc)

        self.exf = copy.deepcopy(self.cnn1[:-1])

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

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(nc, 1),
            torch.nn.LeakyReLU(negative_slope=1.0, inplace=True)
        ).to(self.dev3, dtype=torch.float32)

    def extraction(self, x):
        x.to(self.dev0,  dtype=torch.float32)
        f = [self.exf[0](x)]
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

        if self.wf:
            f =  self.extraction(x)
        if not self.training:
            z = z.detach()

        if self.wf:
            return z,f
        else:
            return z

    def critic(self,X):
        X = self.forward(X)
        z =  torch.reshape(X,(-1,1))
        return self.features_to_prob(z)