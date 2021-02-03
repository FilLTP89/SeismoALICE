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

class DCGAN_DxModelParallele(object):
    """docstring for DCGAN_Dx"""
    def __init__(self, *arg, **kwargs):
        super(DCGAN_DxModelParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DxByGPU(ngpu, nc, ncl, ndf, nly,act,  fpd=0,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.25,
                 n_extra_layers=0,isize=256):
        classname = 'DCGAN_Dx_' + str(ngpu)+'GPU'
        #this following code is equivalent to calls the class it self. 
        """
        Here we define a methode, in which the purpose is to call and return a class by the name 
        This is made in the purpose of spliting the whole network in the GPUs allowed. There for 
        This suppose that one node of ngpu is present in the configuraton of the environement. 
        """  
        module_name = "dcgan_dxmodelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module,classname)

        return class_(ngpu = ngpu, isize = isize, nc = nc, ncl = ncl, ndf = ndf, fpd = fpd, act=act,\
                        nly = nly, ker=ker ,std=std, pad=pad, dil=dil, grp=grp, bn=bn, wf = wf, dpc=dpc,\
                        n_extra_layers = n_extra_layers)


class BasicDCGAN_Dx(Module):
    """docstring for BasicDCGAN_Dx"""
    def __init__(self, *arg,**kwargs):
        super(BasicDCGAN_Dx, self).__init__()
        #ordinal attribute of the GPUs
        self.dev0 = 0
        self.dev1 = 1
        self.dev2 = 2
        self.dev3 = 3

        #initlialization of the GPUs
        self.cnn1 = []
        self.cnn2 = []
        self.cnn3 = []
        self.cnn4 = []

        #trainings
        self.training = True



    def lout(self, nz, nly, increment):
        #Here we specify the logic of the  in_channels/out_channels
        n = nz*2**(increment)
        limit = 512
        #we force the last of the out_channels to not be greater than 512
        val = n if (n<limit or increment<nly) else limit
        return val if val <= limit else limit

        
    def critic(self,X):
        pass


class DCGAN_Dx_1GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_1GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly, act, fpd=1, isize=256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250,
                 n_extra_layers=0):
        super(DCGAN_Dx_1GPU, self).__init__()
        
        in_channels =  nc

        #activation code
        activation = T.activation(act, nly)
        for i in range(1, nly+1):
            out_channels =  self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,dil =dil,\
                    bn=True,dpc=0.0,wn=False)
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 1),
            torch.nn.Sigmoid()
        ).to(self.dev0, dtype=torch.float32)

    def forward(self,x):
        x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        # torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x

    def critc(self,X):
        X = forward(X)
        return self.features_to_prob(X)



class DCGAN_Dx_2GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_2GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, fpd=1,isize=256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250,
                 n_extra_layers=0):
        super(DCGAN_Dx_2GPU, self).__init__()
        self.ngpu= ngpu


        #activation code
        activation = T.activation(act, nly)

        #part I in the GPU0
        in_channels   = nc
        for i in range(1, nly//2+1):
            
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            out_channels = self.lout(ndf, nly, i)
            #The last activation function shall be a sigmoid function
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,ker=ker,std=std,pad=pad,dil =dil,bn=True,dpc=dpc)
            in_channels = out_channels

        """
        The second loop generate the second network, which should be saved after in the GPU1
        """
        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,dil =dil,\
                    bn=True,dpc=0.0,wn=False)            
            in_channels = out_channels 

        """
        Here we define put the network and the GPUs
        we precise that the type will be in float32
        non_blocking = True to free memory when it is not needed anymore
        """
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 1),
            torch.nn.Sigmoid()
        ).to(self.dev1, dtype=torch.float32)

    def forward(self, x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        
        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)
        # torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
            return x
        else:
            return x

    def critc(self,X):
        X = forward(X)
        return self.features_to_prob(X)


class DCGAN_Dx_3GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_3GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, fpd=1, isize=256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250,
                 n_extra_layers=0):
        super(DCGAN_Dx_3GPU, self).__init__()
        

        #activation code
        activation = T.activation(act, nly)
        in_channels = nc
        for i in range(1, nly//3+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act,ker=ker,std=std,pad=pad,dil=dil, bn=True,dpc=dpc)
            in_channels = out_channels


        for i in range(nly//3+1, 2*nly//3+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels, out_channels, act,ker=ker,std=std,pad=pad,dil =dil, bn=True,dpc=dpc)
            in_channels = out_channels

        for i in range(2*nly//3+1, nly+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn3 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,dil =dil,\
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
        # torch.cuda.empty_cache()
        if not self.training:
            x = x.detach()
        return x

class DCGAN_Dx_4GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_4GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, fpd=1, isize=256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.25,
                 n_extra_layers=0):
        super(DCGAN_Dx_4GPU, self).__init__()

        #activation code
        activation = T.activation(act, nly)

        #initialization of the channel
        in_channels =  nc

        #Part I in GPU0
        for i in range(1, nly//4+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn1 += cnn1d(in_channels, out_channels, act, ker=ker,std=std,pad=pad,dil =dil,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part II in GPU1
        for i in range(nly//4+1, 2*nly//4+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn2 += cnn1d(in_channels, out_channels, act,ker=ker,std=std,pad=pad,dil =dil,bn=True,dpc=dpc)
            in_channels = out_channels
        #Part III in GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn3 += cnn1d(in_channels, out_channels, act,ker=ker,std=std,pad=pad,dil =dil,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part IV in GPU3
        for i in range(3*nly//4+1, nly+1):
            out_channels = self.lout(ndf, nly, i)
            act = activation[i-1]
            self.cnn4 += cnn1d(in_channels,out_channels, act,ker=ker,std=std,pad=pad,dil =dil,\
                    bn=True,dpc=0.0,wn=False)            
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2, dtype=torch.float32)

        self.cnn4 = sqn(*self.cnn4)
        self.cnn4.to(self.dev3, dtype=torch.float32)


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
            x = x.dexietach()
        return x