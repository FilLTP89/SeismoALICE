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


class DCGAN_DXZDataParallele(object):
    """docstring for DCGAN_DXZDataParallele"""
    def __init__(self, *args, **kwargs):
        super(DCGAN_DXZDataParallele, self).__init__()
        pass
        
    @staticmethod
    def getDCGAN_DXZDataParallele(name, ngpu, nly, channel, act, path, nc=1024,\
        ker=2,std=2,pad=0, dil=0,grp=0,limit = 256,\
        bn=True,wf=False, dpc=0.25, n_extra_layers= 0, bias = False, *args, **kwargs):

        if name is not None:
            classname = 'DCGAN_DXZ'+name
            #preparation for other DataParallele Class
            try:
                module_name = "dcgan_dxzdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)
                return class_(ngpu=ngpu, nc=nc, nly=nly, channel = channel,\
                                ker=ker,std=std,pad=pad, dil=dil, act = act,\
                                grp=grp, bn=bn, wf=wf, dpc=dpc,path=path,\
                                n_extra_layers=n_extra_layers, limit = limit, bias= bias)
            except Exception as e:
                raise e
                print("The class ", classname, " does not exit")
        else:
            return DCGAN_DXZ(ngpu=ngpu, nc=nc, nly=nly, path = path, channel = channel, act = act,\
                 ker=ker,std=std,pad=pad, dil=dil, grp=grp,\
                 bn=bn, wf=wf, dpc=dpc,\
                 n_extra_layers=n_extra_layers, limit = limit, bias=bias)

class BasicDCGAN_DXZDataParallele(Module):
    """docstring for BasicDCGAN_DXZDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DXZDataParallele, self).__init__()
        self.training = True
        
    def lout(self, nc, nly, increment, limit):
        val =  nc if increment < nly else 1
        return val if val <= limit else limit

    def critic(self,x):
        pass

    def extraction(self,x):
        pass

class DCGAN_DXZ(BasicDCGAN_DXZDataParallele):
    """docstring for DCGAN_DXZ"""
    def __init__(self,ngpu, nly, channel, act, nc=1024,\
        ker=2,std=2,pad=0, dil=1,grp=1, path='',\
        bn=True,wf=False, dpc=0.25, limit =1024,\
        n_extra_layers= 0, bias=False, *args, **kwargs):
        super(DCGAN_DXZ, self).__init__()
        self.ngpu =  ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        self.net = []
        #activation 
        activation = T.activation(act,nly)
        device = tdev("cuda" if torch.cuda.is_available() else "cpu")
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
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
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

        self.cnn1.to(device)
        self.exf.to(device)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(nc, 1),
            torch.nn.LeakyReLU(negative_slope=1.0, inplace=True)
        ).to(device)

    def extraction(self, x):
        f = [self.exf[0](x)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            # z = pll(self.cnn,X,self.gang)
            z = T._forward(X, self.cnn1, self.gang)
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.cnn1(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z

    def critic(self,X):
        X = self.forward(X)
        z =  torch.reshape(X,(-1,1))
        return pll(self.features_to_prob(z))