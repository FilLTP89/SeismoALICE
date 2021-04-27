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
import copy

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
    def getEncoderByGPU(ngpu,dev, nz, nch, ndf, nly, ker, std, pad, act, dil,\
                 channel, grp=1,bn=True,\
                 dpc=0.0,limit = 256, path="", dconv = "",\
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

        return class_(ngpu = ngpu, dev =dev, nz = nz, nch = nch,path=path,dconv = dconv,\
        nly = nly, ndf=ndf, ker = ker, std =std, pad = pad, act=act, dil = dil, grp =grp, bn = bn,\
        dpc=dpc, with_noise = with_noise, dtm=dtm, ffr =ffr, wpc = wpc,limit = limit, channel= channel)

class BasicEncoderModelParallele(Module):
    def __init__(self):
        super(BasicEncoderModelParallele,self).__init__()

        #v)ariable to call frozen model
        self.model = None
        self.splits = 1

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
        self.training = True

    def lout(self,lin, std, ker, pad, dil):
            _out = (lin + 2 * pad - dil*(ker-1)-1)/std + 1
            return int(_out)  

    def cout(self,nz,nch, nly, increment,limit):
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

    def pad(self, nly, incremement, pad, lin, lout, dil, ker, std):
            if incremement == nly:
                _pad = ((lout - 1)*std + dil*(ker - 1) - lin)/2
                return round(_pad)
            else:
                return pad


class Encoder_1GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, channel,ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,limit = 256,path="",dconv = "",\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_1GPU, self).__init__()
        
        self.ngpu= ngpu
        acts = T.activation(act, nly)
        # pdb.set_trace()

        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False

        if dconv:
            _dconv = DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()
        # pdb.set_trace()
        # lin = 4096
        for i in range(1, nly+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            # lin = lout
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 += cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil = dil[i-1], dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 += cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                _bn  = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc 
                self.cnn1 += cnn1d(channel[i-1], channel[i], acts[i-1], ker=ker[i-1],\
                    std=std[i-1],pad=pad[i-1], dil=dil[i-1], bn=_bn, dpc=_dpc, wn=False)
        
        if dconv:
            self.cnn1 = self.cnn1 + _dconv

        self.cnn1  = sqn(*self.cnn1)
        # pdb.set_trace()
        if path:
            self.model.cnn1[-1] = copy.deepcopy(self.cnn1)
            self.cnn1 = self.model
        self.cnn1.to(self.dev0,dtype=torch.float32)

    def forward(self,x):
        x = T._forward_1G(x,self.cnn1)
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
        torch.cuda.empty_cache()
        # x = x.to(self.dev0,dtype=torch.float32)
        # x = self.cnn1(x)
        if not self.training:
            x = x.detach()
        return x
        

class Encoder_2GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, channel, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,limit = 256,path="",dconv = "",\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_2GPU, self).__init__()
        self.ngpu= ngpu
        self.splits = 5
        acts = T.activation(act, nly)
        
        if dconv:
            _dconv = DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        # lin = 4096
        #Part I in the GPU0
        for i in range(1, nly//2+1):
            
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(channel[i-1],channel[i],acts[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(channel[i-1],channel[i],acts[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normally
            else:
                self.cnn1 += cnn1d(channel[i-1],channel[i],acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=dpc,wn=False)
                

        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            _bn  = False if i == nly else bn
            _dpc = 0.0 if i == nly else dpc 
            self.cnn2 += cnn1d(channel[i-1],channel[i], acts[i-1], ker=ker[i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1],\
                        bn=_bn,dpc=_dpc,wn=False)
            
        if dconv:
            self.cnn2 = self.cnn2 + _dconv

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)

    def forward(self,x):
        x = x.to(self.dev0,dtype=torch.float32)
        x = T._forward_2G(x, self.cnn1, self.cnn2)

        if not self.training:
            x = x.detach()
            return x
        else:
            return x

class Encoder_3GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil, channel, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,limit = 256,path="",dconv = "",\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_3GPU, self).__init__()
        self.ngpu = ngpu
        
        #initialization
        acts = T.activation(act, nly)

        if dconv:
            _dconv = DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        #Part I in the GPU0
        for i in range(1, nly//3+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            if i == 1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(channel[i-1],channel[i],acts[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(channel[i-1],channel[i],acts[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                
                self.cnn1 += cnn1d(channel[i-1],channel[i],acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=dpc,wn=False)
                

        #Part II in the GPU1
        for i in range(nly//3+1, 2*nly//3+1):
            
            self.cnn2 += cnn1d(channel[i-1],channel[i],acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=dpc,wn=False)
            

        #Part III in the GPU2
        for i in range(2*nly//3+1, nly+1):
            
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            _bn  = False if i == nly else bn
            _dpc = 0.0 if i == nly else dpc 
            self.cnn3 += cnn1d(channel[i-1],channel[i], acts[i-1], ker=ker[i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1],\
                        bn=_bn,dpc=_dpc,wn=False)
        
        if _dconv:
            self.cnn3 =  self.cnn3 + _dconv
        
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

        if not self.training:
            x = x.detach()
        torch.cuda.empty_cache()
        return x


class Encoder_4GPU(BasicEncoderModelParallele):
    def __init__(self,ngpu,dev, nz,nch,ndf,nly, act,dil,channel, ker=2,std=2,pad=0,grp=1,bn=True,\
                 dpc=0.10,limit = 256,path="",dconv = "",\
                 with_noise=True,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder_4GPU, self).__init__()
        self.ngpu = ngpu
        in_channels = nz
        out_channels = 0
        # lin = 4096
        #initialization
        acts = T.activation(act, nly)
        if dconv:
            _dconv = DConv_62(last_channel = channel[-1], bn = False, dpc = 0.0).network()

        #Part I in the GPU0
        for i in range(1, nly//4+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            if i == 1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = cnn1d(channel[i-1],channel[i],acts[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dpc=dpc,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,\
                    dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = cnn1d(channel[i-1],channel[i],acts[0],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                
                self.cnn1 += cnn1d(channel[i-1],channel[i],acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1],bn=bn,dpc=dpc,wn=False)
                

        #Part II in the GPU1
        for i in range(nly//4+1, 2*nly//4+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            
            self.cnn2 += cnn1d(channel[i-1],channel[i],acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=dpc,wn=False)
            

        #Part III in the GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            self.cnn3 += cnn1d(channel[i-1],channel[i],acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=dpc,wn=False)
            

        #Part III in the GPU3
        for i in range(3*nly//4+1, nly+1):
            
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            _bn = False if i == nly else bn
            _dpc = 0.0 if i == nly else dpc 
            self.cnn4 += cnn1d(channel[i-1],channel[i], acts[i-1], ker=ker[i-1],std=std[i-1],pad=pad[i-1],dil=dil[i-1],\
                        bn=_bn,dpc=_dpc,wn=False)
        if dconv:
            self.cnn4 = self.cnn4 + _dconv    
        
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

        if not self.training:
            x = x.detach()
        torch.cuda.empty_cache()
        return x

