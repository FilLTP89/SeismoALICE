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
from dconv import DConv_62

class EncoderDataParallele(object):
    """docstring for EncoderDataParallele"""
    def __init__(self, *arg, **kwargs):
        super(EncoderDataParallele, self).__init__()
        pass
        
    @staticmethod
    def getEncoder(name, ngpu, dev, nz, nch, ndf, nly, ker, std, config,\
                pad, dil, channel, act,limit,path, dconv, *args, **kwargs):
        
        if name is not None:
            classname = 'Encoder_'+name
            try:
                return type(classname, (BasicEncoderDataParallele, ), dict(ngpu = ngpu,dev =dev, nz = nz, nch = nch, act=act,dconv = dconv,
                        nly = nly, config = config, ker = ker, ndf=ndf,std =std, path=path, pad = pad, dil = dil, channel=channel, limit = limit))
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return Encoder(ngpu = ngpu,dev =dev, ndf=ndf, nz = nz, nch = nch, act=act, dconv = dconv,
                        nly = nly, config = config, ker = ker, std =std, pad = pad, path=path, dil = dil, channel=channel, limit = limit)

class BasicEncoderDataParallele(Module):
    """docstring for BasicEncoderDataParallele"""
    def __init__(self):
        super(BasicEncoderDataParallele, self).__init__()
        self.training = True
        self.model = None
        self.resnet = []
        self.cnn1 = []
        self.branch_broadband = []
        self.branch_filtered = []
        

    def lout(self,nz,nch, nly, increment,limit):
        """
        This code is for conv1d made according to the rule of Pytorch.
        One multiply nz by 2 ^(increment - 1). 
        If, by example, nly 8. we strat from nz^(0) to nz^(6). we stop witnz
        
        """
        limit = 256
        n = increment - 1
        val = int(nz*2**n) if n <= (nly - 2) else nz
        return val if val <= limit else limit
    
class Encoder(BasicEncoderDataParallele):
    """docstring for Encoder"""
    def __init__(self, ngpu,dev,nz,nch,ndf,act,channel,\
                 nly, config,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256, path='',dconv = "",\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2):
        # pdb.set_trace()
        super(Encoder, self).__init__()
        self.ngpu= ngpu
        self.gang = range(ngpu)
        
        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")

        acts = T.activation(act, nly)
        # pdb.set_trace()
        pad
        resnet = True
        if dconv:
            _dconv = DConv_62(last_channel = channel[-1], bn = True, dpc = 0.0).network()
        
        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False
        
        #broadband part
        channel_bb  = config["broadband"]["channel"]
        ker_bb      = config["broadband"]["kernel"]
        std_bb      = config["broadband"]["strides"]
        dil_bb      = config["broadband"]["dilation"]
        pad_bb      = config["broadband"]["padding"]
        nly_bb      = config["broadband"]["nlayers"]
        acts_bb     = T.activation(config["broadband"]["act"], config["broadband"]["nlayers"])

        #filtered part
        channel_fl  = config["filtered"]["channel"]
        ker_fl      = config["filtered"]["kernel"]
        std_fl      = config["filtered"]["strides"]
        pad_fl      = config["filtered"]["padding"]
        nly_fl      = config["filtered"]["nlayers"]
        dil_fl      = config["filtered"]["dilation"]
        acts_fl     = T.activation(config["filtered"]["act"], config["filtered"]["nlayers"])
     
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
        
        for n in range(1,nly_bb+1):
            _bn = True if n<= nly_bb else False
            self.branch_broadband += cnn1d(channel_bb[n-1], channel_bb[n], acts_bb[n-1], ker=ker_bb[n-1],\
                    std=std_bb[n-1],pad=pad_bb[n-1], dil=dil_bb[n-1], bn=_bn, dpc=_dpc, wn=False)

        for n in range(1, nly_fl+1):
            _bn = True if n<= nly_bb else False
            self.branch_filtered += cnn1d(channel_fl[n-1], channel_fl[n], acts_fl[n-1], ker=ker_fl[n-1],\
                    std=std_fl[n-1],pad=pad_fl[n-1], dil=dil_fl[n-1], bn=_bn, dpc=_dpc, wn=False)

        #append the dilated convolutional network to the network
        # pdb.set_trace()
        if dconv:
            self.cnn1  = self.cnn1 + _dconv 

        # self.branch_broadband = sqn(*self.branch_broadband)
        # self.branch_filtered = sqn(*self.branch_filtered)
        # if resnet:
        #     self.resnet = ResNetLayer(channel[-1],channel[-1]).to(device)
        # self.cnn1 += [Flatten()]
        # self.cnn1 += DenseBlock(channel[-1]*opt.batchSize*opt.imatgeSize,opt.nzd) 
        self.cnn_common  = sqn(*self.cnn1)
        self.zy = sqn(*(self.branch_broadband))
        self.zx = sqn(*(self.branch_filtered))
        # pdb.set_trace()
        if path:
            self.model.cnn1[-1] = copy.deepcopy(self.cnn1)
            self.cnn1 = self.model
        self.cnn_common.to(self.device)
        self.zy.to(self.device)
        self.zx.to(self.device)

    def forward(self,x):
        if x.is_cuda and self.ngpu >=1:
            z  = self.cnn_common(x)
            zy =  self.zy(z)
            zx = self.zx(z)
        else:
            z  = self.cnn_common(x)
            zy =  self.zy(z)
            zx = self.zx(z)
        if not self.training:
            zy, zx = zy.detach(), zx.detach()
        return zy, zx