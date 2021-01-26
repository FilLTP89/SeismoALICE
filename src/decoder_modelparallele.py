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

class DecoderModelParallele(object):
    """docstring for EncoderModelParallele"""
    def __init__(self,*arg, **kwargs):
        super(DecoderModelParallele, self).__init__()
        pass
        
    # this methode call the Encoder class by name.
    @staticmethod
    def getDecoderByGPU(ngpu,nz,nch,ndf,nly,act,dil,\
                 ker=2,std=2,pad=0,opd=0,grp=1,dpc=0.10):
        classname = 'Decoder_' + str(ngpu)+'GPU'
        #this following code is equivalent to calls the class it self. 
        """
        Here we define a methode, in which the purpose is to call and return a class by the name 
        This is made in the purpose of spliting the whole network in the GPUs allowed. There for 
        This suppose that one node of ngpu is present in the configuraton of the environement. 
        """  
        module_name = "decoder_modelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module,classname)
        
        return class_(ngpu = ngpu, nz = nz, nch = nch,\
        nly = nly, act=act, ndf =ndf, ker = ker, std =std, pad = pad, opd = opd, grp=grp, dil=dil, dpc = dpc)

class BasicDecoderModelParallele(Module):
    """ Basic Encoder for the GPU classes"""
    def __init__(self):
        super(BasicDecoderModelParallele, self).__init__()
        self.dev0 = 0
        self.dev1 = 1
        self.dev2 = 2
        self.dev3 = 3
        """
        The CNN code will be split in two parts of the GPUs
        """
        #initialization of the cnns:
        self.cnn1 = []
        self.cnn2 = []
        self.cnn3 = []
        self.cnn4 = []

        self.training = True

    def lout(self,nz, nch, nly, increment):
        """
        This code is for convTranspose1d made according to the rule of Pytorch. See official reference :
        One  multiply nz by  2 ^ (nly -incremement -1). 
        if nly 5. we strate from nzd*2^(3), nzd*2^(2), nz*2^(1), nzd*2^(0), nch. 
        Here nch should be equal to 3. 
        Therefore the convolutionnal network strat form :
         (nz,nzd*2^(3)) --> (nzd*2^(3),nzd*2^(2)) --> (nzd*2^(2),nzd*2^(1))
         --> (nzd*2^(1),nzd*2^(0))--> (nzd*2^(0),nch)
        """
        nzd = nz
        n = nly-2-increment+1
        return nzd*2**n if n >=0 else nch   

class Decoder_1GPU(BasicDecoderModelParallele):
    def __init__(self,ngpu,nz,nch,ndf,nly,act,\
                 ker=7,std=4,pad=0,opd=0,dil=0,grp=1,dpc=0.10):
        super(Decoder_1GPU, self).__init__()
        """
        In this class our intent is the generate the network and after split this latter in
        the whole GPUs allowed. Here this class is if we got only one GPU
        """
        self.ngpu= ngpu
        #initializaton of the cnn
        self.cnn = []
        acts = T.activation(act, nly)
        in_channels   = nz*2
        for i in range(1, nly+1):
            out_channels = self.lout(nz, nch, nly, i)
            """
            This is made in the respectful of the pytorch documentation  :
            See the reference : 
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

            """
            self.cnn1 += cnn1dt(in_channels,out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil, opd=opd, bn=True,dpc=0.0)
            in_channels = out_channels
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0, dtype=torch.float32)
        x = self.cnn1(x)
        # torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x

class Decoder_2GPU(BasicDecoderModelParallele) :
    def __init__(self,ngpu,nz,nch,ndf,nly,act,\
                 ker=7,std=4,pad=0,opd=0,dil=0,grp=1,dpc=0.10):
        super(Decoder_2GPU, self).__init__()
        self.ngpu= ngpu
        acts = T.activation(act, nly)

        """
        The CNN code will be split in two parts of the GPUs
        THe first loop generate the first part of the network, 
        which will be saved in the GPU0 . 
        """
        #part I in the GPU0
        in_channels   = nz*2
        for i in range(1, nly//2+1):
            
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            out_channels = self.lout(nz, nch,nly, i)
            self.cnn1 += cnn1dt(in_channels, out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd,bn=True,dpc=dpc)
            in_channels = out_channels

        """
        The second loop generate the second network, which should be saved after in the GPU1
        """
        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            out_channels = self.lout(nz, nch,nly, i)
            self.cnn2 += cnn1dt(in_channels,out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd, bn=True,dpc=0.0)
            in_channels = out_channels 

        """
        Here we define put the network and the GPUs
        we precise that the type will be in float32
        non_blocking = True to free memory when it is not needed anymore
        """
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0,dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1,dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)

        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)

        torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x


class Decoder_3GPU(BasicDecoderModelParallele) :
    def __init__(self,ngpu,nz,nch,ndf,nly,act,\
                 ker=7,std=4,pad=0,opd=0,dil=0,grp=1,dpc=0.10):
        super(Decoder_3GPU, self).__init__()
        self.ngpu= ngpu

        acts = T.activation(act, nly)

        #part I in the GPU0
        in_channels   = nz*2
        for i in range(1, nly//3+1):
            out_channels = self.lout(nz, nch,nly, i)
            self.cnn1 += cnn1dt(in_channels, out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part II in the GPU1
        for i in range(nly//3+1, 2*nly//3+1):
            out_channels = self.lout(nz, nch, nly, i)
            self.cnn2 += cnn1dt(in_channels, out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part III in the GPU2
        for i in range(2*nly//3+1, nly+1):
            out_channels = self.lout(nz, nch, nly, i)
            self.cnn3 += cnn1dt(in_channels,out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd, bn=True,dpc=0.0)
            in_channels = out_channels

        """
        Here we define put the network and the GPUs
        we precise that the type will be in float32
        non_blocking = True to free memory when it is not needed anymore
        """
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0,dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1,dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2,dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)

        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)

        x = x.to(self.dev2,dtype=torch.float32)
        x = self.cnn3(x)
        
        torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x

class Decoder_4GPU(BasicDecoderModelParallele) :
    def __init__(self,ngpu,nz,nch,ndf,nly,act,\
                 ker=7,std=4,pad=0,opd=0,dil=0,grp=1,dpc=0.10):
        super(Decoder_4GPU, self).__init__()
        self.ngpu= ngpu
        

        acts = T.activation(act, nly)
        
        #part I in the GPU0
        in_channels   = nz*2
        for i in range(1, nly//4+1):
            out_channels = self.lout(nz, nch,nly,i)
            self.cnn1 += cnn1dt(in_channels, out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part II in the GPU1
        for i in range(nly//4+1, 2*nly//4+1):
            out_channels = self.lout(nz, nch,nly,i)
            self.cnn2 += cnn1dt(in_channels, out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part III in the GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            out_channels = self.lout(nz, nch,nly,i)
            self.cnn3 += cnn1dt(in_channels, out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd,bn=True,dpc=dpc)
            in_channels = out_channels

        #Part IV in the GPU4
        for i in range(3*nly//4+1, 4*nly//4+1):
            out_channels = self.lout(nz, nch,nly,i)
            self.cnn4 += cnn1dt(in_channels,out_channels, acts[i-1],ker=ker,std=std,pad=pad,dil =dil,opd=opd, bn=True,dpc=0.0)
            in_channels = out_channels
        """
        Here we define put the network and the GPUs
        we precise that the type will be in float32
        non_blocking = True to free memory when it is not needed anymore
        """
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0,dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1,dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2, dtype=torch.float32)

        self.cnn4 = sqn(*self.cnn4)
        self.cnn4.to(self.dev3, dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)

        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)

        x = x.to(self.dev2,dtype=torch.float32)
        x = self.cnn3(x)

        x = x.to(self.dev3,dtype=torch.float32)
        x = self.cnn4(x)

        torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        return x