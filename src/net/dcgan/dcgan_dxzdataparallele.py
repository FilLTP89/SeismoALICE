from torch.nn.modules import activation
from core.net.basic_model import BasicModel
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
import copy
import importlib


class DCGAN_DXZDataParallele(object):
    """docstring for DCGAN_DXZDataParallele"""
    def __init__(self, *args, **kwargs):
        super(DCGAN_DXZDataParallele, self).__init__()
        # pass
        
    @staticmethod
    def getDCGAN_DXZDataParallele(name, ngpu, nly, channel, act, path, nc=1024,\
        ker=2,std=2,pad=0, dil=0,grp=0,limit = 256, extra =  128, batch_size=128,\
        bn=True,wf=False, dpc=0.25, n_extra_layers= 0, bias = False, prob = False, *args, **kwargs):

        if name is not None:
            classname = 'DCGAN_DXZ_'+name
            #preparation for other DataParallele Class
            try:
                module_name = "net.dcgan.dcgan_dxzdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)
                return class_(ngpu=ngpu, nc=nc, nly=nly, channel = channel,\
                                ker=ker,std=std,pad=pad, dil=dil, act = act, extra=extra,\
                                grp=grp, bn=bn, wf=wf, dpc=dpc,path=path, prob=prob,\
                                n_extra_layers=n_extra_layers, limit = limit, bias= bias, *args, **kwargs)
            except Exception as e:
                raise e
                print("The class ", classname, " does not exit")
        else:
            return DCGAN_DXZ(ngpu=ngpu, nc=nc, nly=nly, path = path, channel = channel, act = act,\
                 ker=ker,std=std,pad=pad, dil=dil, grp=grp, extra=extra,\
                 bn=bn, wf=wf, dpc=dpc, prob = prob,\
                 n_extra_layers=n_extra_layers, limit = limit, bias=bias, *args, **kwargs)

class BasicDCGAN_DXZDataParallele(BasicModel):
    """docstring for BasicDCGAN_DXZDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DXZDataParallele, self).__init__()
        self.training = True
        self.net   = []
        self.final = []
        
    def lout(self,nch,padding, dilation, kernel_size, stride):
        """
        This code is for conv1d made according to the rule of Pytorch.
        One multiply nz by 2 ^(increment - 1). 
        If, by example, nly 8. we strat from nz^(0) to nz^(6). we stop witnz
        
        """
        lin = nch
        for pad, dil, ker, std in zip(padding, dilation, kernel_size, stride):
            lin = int((lin + 2* pad - dil*(ker-1)-1)/std + 1)

        return lin

    def critic(self,x):
        pass

    def extraction(self,x):
        pass

    def block_conv(self, channel, kernel, strides, dilation, padding, dpc, activation, *args, **kwargs):
        cnn  = []
        pack = zip(channel[:-1], channel[1:], kernel, strides, dilation, padding, activation)
        for in_channels, out_channels, kernel_size, stride, padding, dilation, acts in pack:
            cnn += cnn1d(in_channels=in_channels, 
                        out_channels=out_channels,
                        ker = kernel_size,
                        std = stride,
                        pad = padding,
                        act=acts,
                        dil = dilation,*args, **kwargs)
        return cnn

    def discriminate_circle(self,x,x_r):
        return self(zcat(x,x)), self(zcat(x,x_r))

class DCGAN_DXZ(BasicDCGAN_DXZDataParallele):
    """docstring for DCGAN_DXZ"""
    def __init__(self,ngpu, nly, channel, act, nc=1024,\
        ker=2,std=2,pad=0, dil=1,grp=1, path='', extra=128,batch_size = 128,\
        bn=True,wf=False, dpc=0.25, limit =1024, prob = False,\
        n_extra_layers= 1, bias=False, *args, **kwargs):
        super(DCGAN_DXZ, self).__init__()
        self.ngpu =  ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        
        #activation 
        # breakpoint()
        activation = T.activation(act,nly)
        device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        # pdb.set_trace()
        if path:
            self.cnn1 = T.load_net(path)
            # Freeze model weights
            for param in self.cnn1.parameters():
                param.requires_grad = False
        #initialisation of the input channel

        lout = self.lout(nch=4096,\
            padding    =pad,\
            dilation   =dil,\
            stride     =std,\
            kernel_size=ker)
        
        for i in range(1, nly+1):
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            
            act = activation[i-1]
            # according to Randford et al.,2016 in the discriminator 
            # batchnormalization is appliyed on all layers with exception for the first layer
            _bn = False if i == 1 else bn
            self.net.append(ConvBlock(ni = channel[i-1], no = channel[i],\
                ker = ker[i-1], std = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = bias,\
                bn = _bn, dpc = dpc, act = act))
            # self.cnn1 += cnn1d(in_channels,nc, act, ker=ker,std=std,pad=pad,\
            #         bn=False,bias=True, dpc=dpc,wn=False)

        for _ in range(1, n_extra_layers+1):
            self.net +=cnn1d(channel[-1],nc, activation[i-1],\
                ker=3, std=1, pad=1, bn=False, bias =bias, dpc=dpc, wn = False)
            self.net += torch.nn.Linear(nc,1)
            self.net += torch.nn.LeakyReLU(negative_slope=1.0, inplace=True)

        # any changes made to a copy of object do not reflect in the original object
        self.exf = copy.deepcopy(self.net[:-1])
        """
            Adding if it needed a final part to improve code
        """
        if prob: 
            self.final = [
                Squeeze(),
                # Shallow(shape = lout*channel[-1]),
                # nn.Linear(lout*channel[-1], channel[-1]),
                # nn.Flatten(1),
                # according to Randford et al., 2016, no more Full connected layer is needed
                # only a flattend is applied in the discriminator
                # torch.nn.Linear(nc, 1),
                torch.nn.Sigmoid()]
        else:
            self.final = []
        self.net = self.net + self.final
        self.net = sqn(*self.net)
        # self.features_to_prob = torch.nn.Sequential(
        #     torch.nn.Linear(channel[-1], 1),
        #     #activation, leakyReLU if WGAN and sigmoid if GAN
        #     activation[-1]
        # )
        # pdb.set_trace()
        if path:
            self.cnn1.cnn1[-1] = copy.deepcopy(self.net[:])
        else: 
            self.cnn1 = copy.deepcopy(self.net) 
        del self.net

        self.exf = sqn(*self.exf)

        # self.cnn1.to(device)
        # self.exf.to(device)

       

    def extraction(self, x):
        f = [self.exf[0](x)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,X):
        z = self.cnn1(X)
        # z = torch.reshape(z,(-1,1))
        # z = self.features_to_prob(z)
        if self.wf:
            f = self.extraction(X) 
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            # print("\tIn Model: input size", X.size(),
            #   "output size", z.size())
            return z

    # def critic(self,X):
    #     X = self.forward(X)
    #     z =  torch.reshape(X,(-1,1))
    #     return pll(self.features_to_prob(z))


        

class DCGAN_DXZ_Flatten(BasicDCGAN_DXZDataParallele):
    """docstring for    DCGAN_Dx"""
    def __init__(self,ngpu, nly, channel, act, nc=1024,\
        ker=2,std=2,pad=0, dil=1,grp=1, path='',extra=128,batch_size = 128,\
        bn=True,wf=False, dpc=0.25, limit =1024, prob = False,\
        n_extra_layers= 1, bias=False, *args, **kwargs):
        super(DCGAN_DXZ_Flatten, self).__init__()

        activation = T.activation(act, nly)
        self.cnn  = []

        lout = self.lout(nch= nc,
                padding     = pad, 
                dilation    = dil, 
                kernel_size = ker, 
                stride      = std)
        
        self.cnn += self.block_conv(
            channel     = channel, 
            kernel      = ker, 
            strides     = std, 
            dilation    = dil, 
            padding     = pad, 
            dpc         = dpc, 
            activation  = activation, 
            bn          = bn,
            normalization = torch.nn.BatchNorm1d
            ) 
        if prob:
            self.cnn +=[
                nn.Flatten(start_dim = 1, end_dim=2),
                # Shallow(shape=(batch_size,lout*channel[-1])),
                Linear(lout*channel[-1],1),
                nn.Sigmoid()
            ]

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self,x):
        z = self.cnn(x)
        if not self.training:
            z = z.detach()
        return z