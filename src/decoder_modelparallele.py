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
from dconv import Transopose_DConv_62

class DecoderModelParallele(object):
    """docstring for EncoderModelParallele"""
    def __init__(self,*arg, **kwargs):
        super(DecoderModelParallele, self).__init__()
        pass
        
    # this methode call the Encoder class by name.
    @staticmethod
    def getDecoderByGPU(ngpu,nz,nch,ndf,nly,act,dil,channel,dconv,\
                 ker=2,std=2,pad=0,opd=0,grp=1,dpc=0.0,path='',limit = 256, bn = True, n_extra_layers=0):
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
        
        return class_(ngpu = ngpu, nz = nz, nch = nch, limit = limit, bn = bn, path=path,\
        nly = nly, act=act, ndf =ndf, ker = ker, std =std, pad = pad, opd = opd,dconv = dconv,\
         grp=grp, dil=dil, dpc = dpc,n_extra_layers=n_extra_layers, channel = channel)

class BasicDecoderModelParallele(Module):
    """ Basic Encoder for the GPU classes"""
    def __init__(self):
        super(BasicDecoderModelParallele, self).__init__()
        self.dev0 = 0
        self.dev1 = 1
        self.dev2 = 2
        self.dev3 = 3

        self.splits = 1
        """
        The CNN code will be split in two parts of the GPUs
        """
        #initialization of the cnns:
        self.cnn1 = []
        self.cnn2 = []
        self.cnn3 = []
        self.cnn4 = []

        self.training = True

    def lout(self,nz, nch, nly, increment,limit):
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
        val = nzd*2**n if n >=0 else nch
        return val if val <= limit else limit

    def kout(self, nly, incremement, ker):
        return ker*2 if incremement <= nly//2 else ker

    def pad(self, nly, incremement, pad):
        return 4 if incremement == 1 else pad



class Decoder_1GPU(BasicDecoderModelParallele):
    def __init__(self,ngpu,nz,nch,ndf,nly,act,channel,dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=0,path='',grp=1,dpc=0.0,limit = 256, bn =  True, n_extra_layers = 0):
        super(Decoder_1GPU, self).__init__()
        """
        In this class our intent is the generate the network and after split this latter in
        the whole GPUs allowed. Here this class is if we got only one GPU
        """
        self.ngpu  = ngpu
        #initializaton of the cnn
        self.cnn   = []
        acts       = T.activation(act, nly)

        # pdb.set_trace()
        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False

        if dconv:
            _dconv = Transopose_DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        # pdb.set_trace()
        for i in range(1, nly+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            """
            This is made in the respectful of the pytorch documentation  :
            See the reference : 
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html

            """
            _dpc = 0.0 if i ==nly else dpc
            _bn =  False if i == nly else bn
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1], opd=opd[i-1], bn=_bn,dpc=_dpc)
            

        for i in range(0,n_extra_layers):
            #adding LeakyReLU activation function
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[0],ker=3,std=1,pad=1,\
              dil =1, opd=0, bn=True, dpc=0.0)

        if dconv:
            self.cnn1 = self.cnn1 + _dconv
        # pdb.set_trace()
        self.cnn1 = sqn(*self.cnn1)
        if path: 
            self.cnn1[-1] = self.model
        self.cnn1.to(self.dev0, dtype=torch.float32)

    def forward(self,x):
        # ret    = []
        # splits = iter(x.split(self.splits, dim = 0))
        # s_next = next(splits)
        # s_prev = self.cnn1(s_next).to(self.dev0)

        # for s_next in splits:
        #     # s_prev = self.cnn1(s_prev)
        #     ret.append(s_prev)
        #     s_prev = self.cnn1(s_next).to(self.dev0)

        # # s_prev = self.cnn1(s_prev)
        # ret.append(s_prev)
        # x = torch.cat(ret).to(self.dev0)
        x = T._forward_1G(x,self.cnn1)
        torch.cuda.empty_cache()
        # x = x.to(self.dev0,dtype=torch.float32)
        # x = self.cnn1(x)
        if not self.training:
            x = x.detach()
        return x

class Decoder_2GPU(BasicDecoderModelParallele) :
    def __init__(self,ngpu,nz,nch,ndf,nly,act,channel,dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=0,path='',grp=1,dpc=0.10,limit = 256, bn = True,n_extra_layers = 0):
        super(Decoder_2GPU, self).__init__()
        self.ngpu= ngpu
        acts = T.activation(act, nly)

        """
        The CNN code will be split in two parts of the GPUs
        THe first loop generate the first part of the network, 
        which will be saved in the GPU0 . 
        """
        if dconv:
            _dconv = Transopose_DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        #part I in the GPU0
        in_channels   = nz*2
        for i in range(1, nly//2+1):
            
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)

            _dpc = 0.0 if i==1 else dpc
            
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1],bn=bn,dpc=_dpc)
            

        """
        The second loop generate the second network, which should be saved after in the GPU1
        """
        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            _dpc = 0.0 if i==nly else dpc
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            _bn =  False if i == nly else bn
            self.cnn2 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1], bn=_bn,dpc=_dpc)
        if dconv:
            self.cnn2 = self.cnn2 + _dconv     

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
        # x = self.cnn1(x)

        # x = x.to(self.dev1,dtype=torch.float32)
        # x = self.cnn2(x)
        x = T._forward_2G(x,self.cnn1,self.cnn2)
        torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        # torch.cuda.empty_cache()
        return x


class Decoder_3GPU(BasicDecoderModelParallele) :
    def __init__(self,ngpu,nz,nch,ndf,nly,act,channel,dconv = "",\
                 ker=7,std=4,pad=0,opd=0,dil=0,path='',grp=1,dpc=0.10,limit = 256,bn=True,n_extra_layers = 0):
        super(Decoder_3GPU, self).__init__()
        self.ngpu= ngpu

        acts = T.activation(act, nly)
        if dconv:
            _dconv = Transopose_DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        #part I in the GPU0
        in_channels   = nz*2
        for i in range(1, nly//3+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1],bn=bn,dpc=dpc)
            

        #Part II in the GPU1
        for i in range(nly//3+1, 2*nly//3+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            self.cnn2 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1],bn=True,dpc=dpc)
            

        #Part III in the GPU2
        for i in range(2*nly//3+1, nly+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            _dpc = 0.0 if i==nly else dpc
            _bn =  False if i == nly else bn
            self.cnn3 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1], bn=_bn, dpc=_dpc)
        
        if dconv:
            self.cnn3 = self.cnn3 + _dconv

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
        
        # torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        torch.cuda.empty_cache()
        return x

class Decoder_4GPU(BasicDecoderModelParallele) :
    def __init__(self,ngpu,nz,nch,ndf,nly,act,\
                 ker=7,std=4,pad=0,opd=0,dil=0,path='',grp=1,dpc=0.10, limit = 256,bn = True,n_extra_layers = 0):
        super(Decoder_4GPU, self).__init__()
        self.ngpu= ngpu
        

        acts = T.activation(act, nly)
        if dconv:
            _dconv = Transopose_DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        
        #part I in the GPU0
        in_channels   = nz*2
        for i in range(1, nly//4+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            self.cnn1 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1],bn=bn,dpc=dpc)
            

        #Part II in the GPU1
        for i in range(nly//4+1, 2*nly//4+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            self.cnn2 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1],bn=bn,dpc=dpc)
            

        #Part III in the GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            out_channels = self.lout(nz, nch,nly,i,limit)
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            self.cnn3 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1],bn=bn,dpc=dpc)
            

        #Part IV in the GPU4
        for i in range(3*nly//4+1, 4*nly//4+1):
            
            _ker = self.kout(nly,i, ker)
            _pad = self.pad(nly, i, pad)
            _dpc = 0.0 if i==nly else dpc
            _bn =  False if i == nly else bn
            self.cnn4 += cnn1dt(channel[i-1],channel[i], acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1],opd=opd[i-1], bn=_bn,dpc=_dpc)
        
        if dconv : 
            self.cnn4 = self.cnn4 + _dconv
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

        # torch.cuda.empty_cache()
        if not self.training:
            x=x.detach()
        torch.cuda.empty_cache()
        return x