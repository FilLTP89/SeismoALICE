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

class EncoderModelParallele(object):
    """
    EncoderModelParallele manage creation of class for working in 
    the Model Parallele algorithme implemented by pytorch
    """

    def __init__(self, *arg, **kwargs):
        super(EncoderModelParallele, self).__init__()
        pass
        
    # this methode call the Encoder class by name.
    @staticmethod
    def getEncoderByGPU(ngpu,dev, nz, nch, ndf, nly, ker, std, pad, act, dil,grp=1,bn=True,\
                 dpc=0.0,limit = 256,\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2):
        """
        We define a method responsible to call an instance of the class by name. 
        The name depend on the GPU of the system. 
        """
        #name of the class based on GPUs
        classname = 'Encoder_' + str(ngpu)+'GPU'
        module_name = "encoder_modelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module,classname)

        return class_(ngpu = ngpu, dev =dev, nz = nz, nch = nch,\
        nly = nly, ndf=ndf, ker = ker, std =std, pad = pad, act=act, dil = dil, grp =grp, bn = bn,\
        dpc=dpc, with_noise = with_noise, dtm=dtm, ffr =ffr, wpc = wpc,limit = limit)

class BasicEncoderModelParallele(Module):
    def __init__(self):
        super(BasicEncoderModelParallele,self).__init__()
        #ordinal values of the GPUs
        self.dev0 = 0
        self.dev1 = 1
        self.dev2 = 2
        self.dev3 = 3

        #initialization of the cnns
        self.cnn1 = []
        self.cnn2 = []
        self.cnn3 = []
        self.cnn4 = []

        #traing
        self.trainig = True



    def lout(self,nz,nch, nly, increment,limit):
        """
        This code is for conv1d made according to the rule of Pytorch.
        One multiply nz by 2 ^(increment - 1). 
        If, by example, nly 8. we strat from nz^(0) to nz^(6). we stop witnz
        
        """
        n = increment - 1
        val = nz*2**n if n <= (nly - 2) else nz
        return val if val<= limit else limit

    def kout(self, nly, incremement, ker):
        return ker if incremement <= nly//2+1 else ker*2

    def pad(self, nly, incremement, pad):
        return 4 if incremement == nly else pad


class Encoder_1GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,limit = 256,\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_1GPU, self).__init__()
        self.ngpu= ngpu
        acts = T.activation(act, nly)
        
        in_channels = nz
        for i in range(1, nly+1):
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(nch*2,nz, acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil, dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(nch*2,nz, acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                out_channels = self.lout(nz,nch,nly,i,limit)
                _bn  = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc 
                self.cnn1 += cnn1d(in_channels,out_channels, acts[i-1], ker=_ker,std=std,pad=_pad,dil =dil,\
                        bn=_bn,dpc=_dpc,wn=False)
                in_channels = out_channels
            
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0,dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        torch.cuda.empty_cache()
        if not self.trainig:
            x = x.detach()
        return x

class Encoder_2GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_2GPU, self).__init__()
        self.ngpu= ngpu
        acts = T.activation(act, nly)
        in_channels = nz 
        #Part I in the GPU0
        for i in range(1, nly//2+1):
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(nch*2,nz,acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(nch*2,nz,acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=0.0,wn=False)
            #else we proceed normally
            else:
                out_channels = self.lout(nz,nch,nly,i,limit)
                self.cnn1 += cnn1d(in_channels, out_channels,acts[i-1],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=dpc,wn=False)
                in_channels = out_channels

        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            out_channels = self.lout(nz,nch,nly,i,limit)
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            _bn  = False if i == nly else bn
            _dpc = 0.0 if i == nly else dpc 
            self.cnn2 += cnn1d(in_channels,out_channels, acts[i-1], ker=_ker,std=std,pad=_pad,dil =dil,\
                        bn=_bn,dpc=_dpc,wn=False)
            in_channels = out_channels

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        x = x.to(self.dev1,dtype=torch.float32)
        x = self.cnn2(x)
        torch.cuda.empty_cache()
        if not self.trainig:
            x = x.detach()
        return x

class Encoder_3GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_3GPU, self).__init__()
        self.ngpu = ngpu
        in_channels = nz
        #initialization
        acts = T.activation(act, nly)
        #Part I in the GPU0
        for i in range(1, nly//3+1):
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            if i == 1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(nch*2,nz,acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(nch*2,nz,acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                out_channels = self.lout(nz,nch,nly,i,limit)
                self.cnn1 += cnn1d(in_channels, out_channels,acts[i-1],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=dpc,wn=False)
                in_channels = out_channels

        #Part II in the GPU1
        for i in range(nly//3+1, 2*nly//3+1):
            out_channels = self.lout(nz,nch,nly,i,limit)
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            self.cnn2 += cnn1d(in_channels, out_channels,acts[i-1],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=dpc,wn=False)
            in_channels = out_channels

        #Part III in the GPU2
        for i in range(2*nly//3+1, nly+1):
            out_channels = self.lout(nz,nch,nly,i,limit)
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            _bn = False if i == nly else bn
            _dpc = 0.0 if i == nly else dpc 
            self.cnn3 += cnn1d(in_channels,out_channels, acts[i-1], ker=_ker,std=std,pad=_pad,dil =dil,\
                        bn=_bn,dpc=_dpc,wn=False)
            in_channels = out_channels
        
        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)
        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = self.cnn1(x)
        x = x.to(self.dev1,dtype=torch.float32)
      
        x = x.to(self.dev2,dtype=torch.float32)
        x = self.cnn3(x)

        torch.cuda.empty_cache()
        if not self.trainig:
            x = x.detach()
        return x


class Encoder_4GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_4GPU, self).__init__()
        self.ngpu = ngpu
        in_channels = nz
        #initialization
        acts = T.activation(act, nly)
        #Part I in the GPU0
        for i in range(1, nly//4+1):
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            if i == 1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(nch*2,nz,acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dpc=dpc,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(nch*2,nz,acts[0],ker=_ker,std=std,pad=_pad,bn=bn,dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                out_channels = self.lout(nz,nch,nly,i,limit)
                self.cnn1 += cnn1d(in_channels, out_channels,acts[i-1],ker=_ker,std=std,pad=_pad,dil =dil,bn=bn,dpc=dpc,wn=False)
                in_channels = out_channels

        #Part II in the GPU1
        for i in range(nly//4+1, 2*nly//4+1):
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            out_channels = self.lout(nz,nch,nly,i,limit)
            self.cnn2 += cnn1d(in_channels, out_channels,acts[i-1],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=dpc,wn=False)
            in_channels = out_channels

        #Part III in the GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            out_channels = self.lout(nz,nch,nly,i,limit)
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            self.cnn3 += cnn1d(in_channels, out_channels,acts[i-1],ker=_ker,std=std,pad=_pad,bn=bn,dil =dil,dpc=dpc,wn=False)
            in_channels = out_channels

        #Part III in the GPU3
        for i in range(3*nly//4+1, nly+1):
            out_channels = self.lout(nz,nch,nly,i,limit)
            _ker = self.kout(nly,i,ker)
            _pad = self.pad(nly,i,pad)
            _bn = False if i == nly else bn
            _dpc = 0.0 if i == nly else dpc 
            self.cnn4 += cnn1d(in_channels,out_channels, acts[i-1], ker=_ker,std=std,pad=_pad,dil =dil,\
                        bn=_bn,dpc=_dpc,wn=False)
            in_channels = out_channels
        
        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)
        self.cnn4 = sqn(*self.cnn4)
        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)
        self.cnn4.to(self.dev3,dtype=torch.float32)

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
        if not self.trainig:
            x = x.detach()
        return x

