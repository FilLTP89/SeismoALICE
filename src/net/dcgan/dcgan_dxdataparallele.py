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
import importlib
from torch import device as tdev


class DCGAN_DxDataParallele(object):
    """docstring for DCGAN_DxDataParallele"""
    def __init__(self, *arg, **kwargs):
        super(DCGAN_DxDataParallele, self).__init__()
        pass
    @staticmethod
    def getDCGAN_DxDataParallele(name, ngpu, nc, ncl, ndf, nly,act,channel, path, fpd=0, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0,
                 n_extra_layers=0,isize=256, prob = False):
        if name is not None:
            classname = 'DCGAN_Dx_'+ name
            #preparation for other DataParallele Class
            try:
                module_name = "net.dcgan.dcgan_dxdataparallele"
                module = importlib.import_module(module_name)
                class_ = getattr(module,classname)
                return class_(ngpu=ngpu, isize=isize, nc=nc, ncl=ncl, ndf=ndf,  channel = channel, fpd=fpd, act=act, nly=nly,\
                 ker=ker,std=std,pad=pad, dil=dil,grp=grp,bn=bn,wf=wf, dpc=dpc, limit = limit,\
                 n_extra_layers=n_extra_layers, path=path, prob=prob)
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return DCGAN_Dx(ngpu = ngpu, isize = isize, nc = nc, ncl = ncl, ndf = ndf, fpd = fpd,channel = channel,\
                        nly = nly, ker=ker ,std=std, pad=pad, dil=dil, grp=grp, bn=bn, wf = wf, dpc=dpc,\
                        n_extra_layers = n_extra_layers,limit = limit,path = path, act = act, prob = prob)

class BasicDCGAN_DxDataParallele(BasicModel):
    """docstring for BasicDCGAN_DxDataParallele"""
    def __init__(self):
        super(BasicDCGAN_DxDataParallele, self).__init__()
        self.training = True
        self.wf       = True
        self.prc      = []
        self.exf      = []
        self.extra    = []
        self.final    = []
        self.cnn1     = []

    def lout(self,nch,padding, dilation, kernel, stride):
        """
        This code is for conv1d made according to the rule of Pytorch.
        One multiply nz by 2 ^(increment - 1). 
        If, by example, nly 8. we strat from nz^(0) to nz^(6). we stop witnz
        
        """
        lin = nch
        for pad, dil, ker, std in zip(padding, dilation, kernel, stride):
            lin = int((lin + 2* pad - dil*(ker-1)-1)/std + 1)

        return lin
    
    def discriminate_circle(self,x,x_rec): 
        return self(zcat(x,x)), self(zcat(x,x_rec))

    def block_linear(self, channel, activation):
        cnn = []
        for in_channels, out_channels, acts in zip(channel[:-1], channel[1:], activation):
            cnn += [nn.Linear(in_channels, out_channels)]
            cnn += [acts]
            cnn += [Dpout(dpc=dpc)]
        return cnn

    def block_conv(self, channel, kernel, strides, dilation, 
                    padding, activation, *args, **kwargs):
        cnn = []
        for in_channels, out_channels, kernel_size,\
            stride, dilation, padding, acts in zip(channel[:-1],\
            channel[1:], kernel, strides, dilation, padding, activation):
            cnn += cnn1d(in_channels=in_channels, 
                        out_channels=out_channels,
                        ker =kernel_size,
                        std = stride,
                        pad = padding,
                        dil = dilation,*args, **kwargs)
        return cnn

class   DCGAN_Dx(BasicDCGAN_DxDataParallele):
    """docstring for    DCGAN_Dx"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel, fpd=1, isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0, path = '', prob = False,
                 n_extra_layers=0):

        super(DCGAN_Dx, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.cnn  = []

        #activation code
        activation = T.activation(act, nly)
        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        
        #extraction features 
        self.wf  = wf
        self.net = []
        # pdb.set_trace()
        if path:
            self.cnn1 = T.load_net(path)
            # Freeze model weights
            for param in self.cnn1.parameters():
                param.requires_grad = False

        #building network
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ker = ker[0], std = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2, nly+1):
            act = activation[i-1]
            # according to Randford et al.,2016 in the discriminator 
            # batchnormalization is appliyed on all layers with exception for the first layer
            _bn = False if i == 1 else bn
            _dpc = 0.0 if i == nly else dpc
            self.net.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ker = ker[i-1], std = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act)) 

        """
            The kernel = 3 and the stride = 1 not change the third dimension
        """
        for _ in range(0,n_extra_layers):
            self.extra+=[Conv1d(in_channels = channel[i],out_channels=channel[i],\
                kernel_size = 3, stride = 1, padding=1, bias=False)]


        if prob:
            self.final = [
                Flatten(),
                # according to Randford et al., 2016, no more Full connected layer is needed
                # only a flattend is applied in the discriminator
                # torch.nn.Linear(nc, 1),
                torch.nn.Sigmoid()]
        else:
            self.final+=[Conv1d(channel[i], channel[i], 3, padding=1, bias=False)]
            #self.final+=[BatchNorm1d(channel[i])]
            self.final+=[Dpout(dpc=dpc)]
            self.final+=[activation[-1]]
        
        #compute values 
        self.exf  = self.net
        self._net = self.prc + self.net + self.extra + self.final
        self._net = sqn(*self._net)
        self.net  = sqn(*self.net)

        #creating sequentially the Network
        
        # pdb.set_trace()
        if path:
            self.cnn1.cnn1[-5] = self.net[:]
        else:   
            self.cnn1 = self._net
        del self.net
        del self._net

        # self.cnn1.to(self.device)

        self.prc = sqn(*self.prc)
        # self.prc.to(self.device)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,z):
        z = self.cnn1(z)
        if self.wf:
            f = self.extraction(z)

        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z


class DCGAN_Dx_Lite(BasicDCGAN_DxDataParallele):
    """docstring for DCGAN_Dx_Lite"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel, fpd=1, isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0, path = '', prob = False,
                 n_extra_layers=0):
        super(DCGAN_Dx_Lite, self).__init__()
        activation = T.activation(act, nly)

        self.cnn = self.block_conv( channel = channel,kernel = ker,\
                    strides = std, dilation= dil,  activation = activation,\
                    padding = pad, bn = bn, dpc = dpc)

        lout = self.lout(nch = 4096,padding = pad, dilation = dil,\
                        kernel = ker, stride = std)

        self.cnn += [Shallow(shape = lout*channel[-1])]
        self.cnn += [nn.Linear(lout*channel[-1],lout)]
        self.cnn += [nn.Linear(lout,channel[-1])]

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self, x):
        return self.cnn(x)
        

class DCGAN_Dx_Flatten(BasicDCGAN_DxDataParallele):
    """docstring for    DCGAN_Dx"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel, fpd=1, isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.0, path = '', prob = False,
                 n_extra_layers=0):
        super(DCGAN_Dx_Flatten,self).__init__()

        activation = T.activation(act, nly)
        self.cnn  = []

        for in_channels, out_channels, acts in zip(channel[:-1], channel[1:], activation):
            self.cnn += [nn.Linear(in_channels, out_channels)]
            self.cnn += [acts]
            self.cnn += [Dpout(dpc=dpc)]

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self,x): 
        return self.cnn(x)



