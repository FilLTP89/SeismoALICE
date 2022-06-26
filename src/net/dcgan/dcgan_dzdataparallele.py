from torch.nn.modules import activation
from core.net.basic_model import BasicModel
from conv.resnet.residual import DecoderResnet,block_2x2
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
import importlib
from torch import device as tdev
import copy

class DCGAN_DzDataParallele(object):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,*args, **kwargs):
        super(DCGAN_DzDataParallele, self).__init__(*args, **kwargs)
        pass

    @staticmethod
    def getDCGAN_DzDataParallele(name,ngpu, nc,nz, act, ncl, ndf, path, nly, channel, bn, prob, fpd=1,\
                 ker=2,std=2,pad=0, dil=0,grp=0,wf=False, dpc=0.0,extra = 128, 
                 n_extra_layers=0, limit = 256, bias = False, batch_size = 128,*args,**kwargs):
        if name is not None:
            classname = 'DCGAN_Dz_' + name
            #prepraring calling class by name if the name exist
            try:
                module_name = "net.dcgan.dcgan_dzdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)

                return class_(ngpu = ngpu, nc=nc, nz =nz, ncl=ncl, ndf=ndf, act=act,fpd=fpd, nly = nly,channel = channel,\
                 ker=ker,std=std,pad=pad, path=path, prob=prob, dil=dil,grp=grp,bn=bn,wf=wf, dpc=dpc, limit = limit,bias = bias,
                 n_extra_layers=n_extra_layers, batch_size=batch_size, extra = extra, *args,**kwargs)
            except Exception as e:
                raise e
                print("The class ", classname, " does not exit")
        else:
            return DCGAN_Dz(ngpu, nc=nc, ncl=ncl, nz=nz, ndf = ndf, act=act,fpd=fpd, nly = nly, channel = channel,extra= extra,\
                 ker=ker,std=std,pad=pad, dil=dil, prob=prob, batch_size=batch_size, path=path, grp=grp, bn=bn,wf=wf, dpc=dpc, limit = limit,bias = bias,\
                 n_extra_layers=n_extra_layers,*args,**kwargs)


class BasicDCGAN_DzDataParallel(BasicModel):
    """docstring for BasicDCGAN_DzDataParallel"""
    def __init__(self,*args, **kwargs):
        super(BasicDCGAN_DzDataParallel, self).__init__(*args, **kwargs)
        self.training =  True
        self.prc    = []
        self.exf    = []
        self.final  = []

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

    def block_conv(self, channel, kernel, strides, dilation, padding, dpc, activation,bn, *args, **kwargs):
        cnn  = []
        #For the first layer we will not use Batchnorm
        _bool_bn    = [True for _ in range(len(channel))]
        _dpc        = [dpc for _ in range(len(channel))]
        _bool_bn[0] = False
        pack = zip(channel[:-1], channel[1:], kernel, strides,dilation, padding, activation,_bool_bn, _dpc)

        for in_channels, out_channels, kernel_size, stride, dilation, padding, acts, _bn, __dpc in pack:
            cnn += cnn1d(in_channels=in_channels, 
                        out_channels=out_channels,
                        ker = kernel_size,
                        std = stride,
                        pad = padding,
                        act = acts,
                        dil = dilation,
                        bn  = _bn,
                        dpc = __dpc,
                        *args, **kwargs)
        return cnn

    def block_linear(self, channels, acts, dpc, bn= False):
        cnn = []
        for in_channels, out_channels, act in zip(channels[:-1], channels[1:], acts):
            cnn += [nn.Linear(in_channels, out_channels, bias=False)]
            cnn += [nn.BatchNorm1d(out_channels)] if bn else []
            cnn += [act]
            cnn += [Dpout(dpc=dpc)]
        return cnn

    def discriminate_circle(self,x, x_r):
        return self(zcat(x,x)), self(zcat(x,x_r)) 

class DCGAN_Dz(BasicDCGAN_DzDataParallel):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,ngpu, nc, ndf, nz, nly, act, channel, bn, fpd=0,ncl = 512,limit = 512,extra = 128,\
                 ker=2,std=2,pad=0, dil=1,grp=1,wf=False, dpc=0.250, bias = False, prob=False, batch_size=128,\
                 n_extra_layers=0, path='',*args,**kwargs):

        super(DCGAN_Dz, self).__init__(*args, **kwargs)

        device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        #activation functions
        activation = T.activation(act, nly)
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf  = wf
        self.net = []
        layers   = []

        if path:
            self.cnn1 = T.load_nBatch
            # Freeze model weights
            for param in self.cnn1.parameters():
                param.requires_grad = False

        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ker = ker[0], std = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2,nly+1):
            act = activation[i-1]
            self.net.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ker = ker[i-1], std = std[i-1], pad = pad[i-1],\
                bias = bias, act = activation[i-1], dpc = dpc, bn = bn))
            # self.cnn1 += cnn1d(in_channels,out_channels, act, ker=ker,std=std,pad=pad,\
            #         bn=_bn,dpc=dpc,wn=False)
            layers.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                    ker = 3, std = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                    dpc = dpc, act = activation[i-1]))

        for k in range(0,n_extra_layers):
            self.net.append(ConvBlock(ni = channel[i-1],no=channel[i],\
                ker = 3, std = 1, pad = 1, dil = 1, bias = False, bn = bn,\
                dpc = dpc, act = activation[k]))


        if prob:
            self.final = [
                Flatten(),
                # torch.nn.Linear(nc, 1),
                torch.nn.Sigmoid()]
        else:
            self.final.append(Conv1d(channel[i], channel[i], kernel_size=3, stride=1, padding=1))
            self.final.append(nn.LeakyReLU(negative_slope=1.0))
        # pdb.set_trace()
        self.exf = layers[:-1]
        self.exf =sqn(*self.exf)
        # self.exf.to(device)

        self.net.append(Dpout(dpc = dpc))
        # self.net.append(activation[-1])
        self.net = self.prc + self.net +  self.final
        self.net =sqn(*self.net)
        
        # pdb.set_trace()

        if path:
            self.cnn1.cnn1[2] = copy.deepcopy(self.net[0:-2])
        else:
            self.cnn1 = copy.deepcopy(self.net)
        del self.net

        # self.cnn1.to(device)

        self.prc = sqn(*self.prc)
        # self.prc.to(device)

    def extraction(self,X):

        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        # pdb.set_trace()
        z = self.cnn1(X)
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

class DCGAN_Dz_Lite(BasicDCGAN_DzDataParallel):
    def __init__(self,ngpu, nc, ndf, nz, nly, act, channel, bn, fpd=0,ncl = 512,limit = 512,extra = 128,\
                 ker=2,std=2,pad=0, dil=1,grp=1,wf=False, dpc=0.250, bias = False, prob=False,batch_size=128,\
                 n_extra_layers=0, path='',*args,**kwargs):
        super(DCGAN_Dz_Lite, self).__init__(*args, **kwargs)
        
        acts      = T.activation(act, nly)
        self.cnn = []
        # self.cnn1 += [UnSqueeze()]
        # self.cnn1 += [Explode(shape = [extra, limit])]
        # self.cnn1 += [nn.BatchNorm1d(extra)]
        # self.cnn1 += [acts[0]]
        normalization =  partial(nn.InstanceNorm1d)
        lout = self.lout(nch= nc,
                padding     = pad, 
                dilation    = dil, 
                kernel_size = ker, 
                stride      = std)
        # pdb.set_trace()
        self.cnn += self.block_conv(channel, 
            kernel      = ker, 
            strides     = std, 
            dilation    = dil, 
            padding     = pad, 
            dpc         = dpc, 
            activation  = acts, 
            bn          = bn, 
            normalization = normalization)
        self.cnn += [nn.Conv1d(in_channels=channel[-1],out_channels=channel[-1],kernel_size = 3, stride = 1, padding=1)]
        # self.cnn += [normalization(channel[-1])]
        # for i in range(1, nly+1):
        #     _dpc = 0.0 if i ==nly else dpc
        #     _bn =  False if i == nly else bn
        #     self.cnn1 += cnn1d(channel[i-1],channel[i],\
        #         acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
        #         dil=dil[i-1], opd=0, bn=_bn,dpc=_dpc)
        if wf:
            self.cnn[-2:]=[
                nn.Flatten(start_dim = 1, end_dim=2),
                nn.Linear(lout*channel[-1],1,bias = True),
                nn.LeakyReLU(negative_slope=1.0, inplace=True)
                # Dpout(dpc = dpc),
                # normalization(1)
            ]
        
        if prob:
            self.cnn+=[nn.Sigmoid()]

        self.cnn = self.cnn
        self.cnn  = sqn(*self.cnn)


    def forward(self, x):
        xr = self.cnn(x)
        if not self.training:
            xr = xr.detach()
        return xr

class DCGAN_Dz_Resnet(BasicDCGAN_DzDataParallel):
    """docstring for DCGAN_Dz_Resnet"""
    def __init__(self,ngpu, nc, ndf, nz, nly, act, channel, bn, fpd=0,ncl = 512,limit = 512,extra = 128,\
                 ker=2,std=2,pad=0, dil=1,grp=1,wf=False, dpc=0.250, bias = False, prob=False,batch_size=128,\
                 n_extra_layers=0, path='',*args,**kwargs):
        super(DCGAN_Dz_Resnet, self).__init__(*args, **kwargs)

        self.cnn = DecoderResnet(in_signals_channels = 16, 
                out_signals_channels=32,
                channels = [128, 64], 
                layers = [2,2], block=block_2x2
        )
        
        if prob:
            lout = self.lout(nch = nc,
                        padding = pad, dilation = dil,\
                        kernel_size = ker, stride = std)
            self.cnn += [nn.Flatten(start_dim = 1, end_dim=2)]
            self.cnn += [nn.Linear(lout*channel[-1],1, bias=False)]
            self.cnn += [nn.Sigmoid()]


class DCGAN_Dz_Flatten(BasicDCGAN_DzDataParallel):
    """docstring for DCGAN_DzDataParallele"""
    def __init__(self,ngpu, nc, ndf, nz, nly, act, channel, bn, fpd=0,ncl = 512,limit = 512,extra = 128,\
                 ker=2,std=2,pad=0, dil=1,grp=1,wf=False, dpc=0.250, bias = False, prob=False,batch_size=128,\
                 n_extra_layers=0, path='',*args,**kwargs):
        super(DCGAN_Dz_Flatten, self).__init__(*args, **kwargs)
        acts        = T.activation(act, nly)
        self.cnn    = []
        self.wf     = wf
        normalization =partial(nn.BatchNorm1d)
        # self.cnn1 += [UnSqueeze()]
        # self.cnn1 += [Explode(shape = [extra, limit])]
        # self.cnn1 += [nn.BatchNorm1d(extra)]
        # self.cnn1 += [acts[0]]

        for i in range(1, nly+1):
            _dpc = 0.0 if i ==nly else dpc
            _bn =  False if i == nly else bn
            self.cnn += cnn1d(channel[i-1],channel[i],\
                acts[i-1],ker=ker[i-1],std=std[i-1],pad=pad[i-1],\
                dil=dil[i-1], bn=_bn,dpc=_dpc, normalization=normalization)
        
        lout = self.lout(nch= nc,
                padding     = pad, 
                dilation    = dil, 
                kernel_size = ker, 
                stride      = std)

        if self.wf:
            self.cnn +=[
                nn.Flatten(start_dim = 1, end_dim=2),
                nn.Linear(lout*channel[-1],extra,bias = True),
                Dpout(dpc = dpc),
                UnSqueeze(1),
                normalization(1)
            ]

        if prob:
            self.final =[
                Squeeze(),
                nn.Sigmoid()
            ]
        else:
            self.final=[]

        self.cnn = self.cnn+self.final
        self.cnn  = sqn(*self.cnn)

    def forward(self,x):
        z = self.cnn(x)
        if not self.training:
            z = z.detach()
        return z


