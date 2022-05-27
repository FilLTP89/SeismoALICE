from torch.nn.modules import activation
from core.net.basic_model import BasicModel
from conv.resnet.residual import EncoderResnet,block_2x2
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
from conv.dconv.dconv import DConv_62
import importlib
from conv.resnet.resnet import ResNetEncoder
from conv.oct_conv.oct_conv import OctaveBatchNormActivation
from conv.oct_conv.oct_conv import OctaveConv
from torchviz import make_dot


class EncoderDataParallele(object):
    """docstring for EncoderDataParallele"""
    def __init__(self, *arg, **kwargs):
        super(EncoderDataParallele, self).__init__()
        pass
        
    @staticmethod
    def getEncoder(name, ngpu, dev, nz, nch, ndf, nly, ker, std, config,\
                pad, dil, channel, act,limit,path, dconv, wf, *args, **kwargs):
        # pdb.set_trace()
        if name is not None:
            classname = 'Encoder_' + name
            module_name = "net.encoder.encoder_dataparallele"
            module = importlib.import_module(module_name)
            class_ = getattr(module,classname)
            try:
                return class_(ngpu = ngpu,dev =dev, ndf=ndf, nz = nz, 
                        nch = nch, act=act, dconv = dconv,
                        nly = nly, config = config, ker = ker, std =std, 
                        pad = pad, path=path, dil = dil, channel=channel, 
                        limit = limit, wf = wf, *args, **kwargs)
            except Exception as e:
                raise e
                print("The class ",classname, " does not exit")
        else:
            return Encoder(ngpu = ngpu,dev =dev, ndf=ndf, nz = nz, 
                            nch = nch, act=act, dconv = dconv,
                            nly = nly, config = config, ker = ker,
                            std =std, pad = pad, path=path, dil = dil,\
                            channel=channel, limit = limit, wf= wf, *args, **kwargs)

class BasicEncoderDataParallele(BasicModel):
    """This class get the basic structure of the Encoders object. 
        in this class we define generic method used in the whole program. 
    """
    def __init__(self,*args, **kwargs):
        super(BasicEncoderDataParallele, self).__init__(*args, **kwargs)
        self.training = True
        self.model    = None
        self.resnet   = []
        self.cnn1     = [] 
        self.branch_broadband = []
        self.branch_filtered  = []
        self.branch_common    = []
        self.wf = False

        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        

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

    def block_conv(self, channel, kernel, strides, dilation, padding, dpc, activation, *args, **kwargs):
        pack = zip(channel[:-1], channel[1:], kernel, strides, dilation, padding, activation)
        for in_channels, out_channels, kernel_size, stride, padding, acts in pack:
            cnn += cnn1d(in_channels=in_channels, 
                        out_channels=out_channels,
                        ker =kernel_size,
                        std = stride,
                        pad = padding,
                        dil = dilation,*args, **kwargs)
        return cnn

    def block_linear(self, in_channels, out_channels, acts):
        for in_channels, out_channels, acts in zip(channel[:-1], channel[1:], activation):
            cnn += [nn.Linear(in_channels, out_channels)]
            cnn += [acts]
            cnn += [Dpout(dpc=dpc)]
        return cnn

    def extraction(self, model, x):
        skip_connections = []
        input = x.detach() + 0.
        for layer in self.model.children():
            x =  layer(x) 
            if layer.__class__.__name__ == 'Conv1d' : 
                skip_connections.append(x)
        return skip_connections



class Encoder(BasicEncoderDataParallele):
    def __init__(self, ngpu,dev,nz,nch,ndf,act,channel,\
                 nly, config,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256, path='',dconv = "",wf = False,\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2,*args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        
        self.ngpu= ngpu
        self.gang = range(ngpu)
        self.wf = wf
        acts = T.activation(act, nly)
        # pdb.set_trace()

        if path:
            self.model = T.load_net(path)
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False

        if dconv:
            _dconv = DConv_62(last_channel = channel[-1], bn = True, dpc = 0.0).network()
        # pdb.set_trace()
        # lin = 4096
        for i in range(1, nly+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            # lin = lout
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = self.cnn1+cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil = dil[i-1], dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = self.cnn1+cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                _bn  = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc 
                self.cnn1 = self.cnn1+cnn1d(channel[i-1], channel[i], acts[i-1], ker=ker[i-1],\
                    std=std[i-1],pad=pad[i-1], dil=dil[i-1], bn=_bn, dpc=_dpc, wn=False)
        
        if dconv:
            self.cnn1 = self.cnn1 + _dconv

        self.cnn1  = sqn(*self.cnn1)
        # pdb.set_trace()
        if path:
            self.model.cnn1[-1] = copy.deepcopy(self.cnn1)
            self.cnn1 = self.model
        # self.cnn1.to(self.device)

    

    def forward(self,x):
        z = self.cnn1(x)

        if not self.training:
            result = z.detach() + 0.
            return result
        else:
            # print(z.requires_grad)
            return z

class Encoder_Unic(BasicEncoderDataParallele):
    """docstring for Encoder"""
    def __init__(self, ngpu,dev,nz,nch,ndf,act,channel,\
                 nly, config,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256, path='',dconv = "",\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2, wf = False, *args, **kwargs):
        # pdb.set_trace()
        super(Encoder_Unic, self).__init__(*args, **kwargs)
        self.ngpu= ngpu
        self.gang = range(ngpu)
        self.wf = wf

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
        dpc_bb      = config["broadband"]["dpc"]
        extra_bb    = config["broadband"]["extra"]
        acts_bb     = T.activation(config["broadband"]["act"], config["broadband"]["nlayers"])

        #common part
        channel_com  = config["common"]["channel"]
        ker_com      = config["common"]["kernel"]
        std_com      = config["common"]["strides"]
        dil_com      = config["common"]["dilation"]
        pad_com      = config["common"]["padding"]
        nly_com      = config["common"]["nlayers"]
        dpc_com      = config["common"]["dpc"]
        extra_com    = config["common"]["extra"]
        acts_com     = T.activation(config["common"]["act"], config["common"]["nlayers"])

        
        lout = self.lout(nch=4096, 
            padding    =pad, 
            dilation   =dil,
            stride     =std,
            kernel_size=ker)

        lout_zy = self.lout(nch=lout, 
            padding    =config["broadband"]["padding"], 
            dilation   =config["broadband"]["dilation"],
            stride     =config["broadband"]["strides"],
            kernel_size=config["broadband"]["kernel"])

        lout_zyx = self.lout(nch=lout, 
            padding     =config["common"]["padding"], 
            dilation    =config["common"]["dilation"],
            stride      =config["common"]["strides"],
            kernel_size =config["common"]["kernel"])

        for i in range(1, nly+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            # lin = lout
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = self.cnn1+cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil = dil[i-1], dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = self.cnn1+cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                _bn  = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc 
                self.cnn1 = self.cnn1+cnn1d(channel[i-1], channel[i], acts[i-1], ker=ker[i-1],\
                    std=std[i-1],pad=pad[i-1], dil=dil[i-1], bn=_bn, dpc=_dpc, wn=False)

        # pdb.set_trace()
        for n in range(1,nly_bb+1):
            _bn  = False if n==nly_bb else bn
            _dpc = 0.0   if n==nly_bb else dpc_bb
            self.branch_broadband += cnn1d(channel_bb[n-1],channel_bb[n],\
                acts_bb[n-1],ker=ker_bb[n-1],std=std_bb[n-1],\
                pad=pad_bb[n-1],bn=_bn,dil=dil_bb[n-1],dpc=_dpc,wn=False)

        for n in range(1,nly_com+1):
            _bn  = False if n==nly_com else bn
            _dpc = 0.0   if n==nly_com else dpc_com
            self.branch_common +=cnn1d(channel_com[n-1],channel_com[n],\
                acts_com[n-1],ker=ker_com[n-1],std=std_com[n-1],\
                pad=pad_com[n-1],bn=_bn,dil=dil_com[n-1],dpc=_dpc,wn=False)

        self.branch_common +=[
            nn.Flatten(start_dim = 1, end_dim=2),
            Linear(lout_zyx*channel_com[-1],128),
            nn.LeakyReLU(1.0,inplace=True)
        ]

        self.branch_broadband +=[
            nn.Flatten(start_dim = 1, end_dim=2),
            Linear(lout_zy*channel_bb[-1],384),
            nn.LeakyReLU(1.0,inplace=True)
        ]
        # self.branch_broadband.append(Squeeze())

        # self.branch_broadband +=[
        #             Shallow(shape=lout_zy*channel_bb[-1]),
        #             Linear(lout_zy*channel_bb[-1],extra_bb, bias=False),
        #         ]
        # self.branch_common += [
        #             Shallow(shape=lout_zyx*channel_com[-1]),
        #             Linear(lout_zyx*channel_com[-1],extra_com, bias=False),
        #     ]
        # self.branch_broadband+=[
        #             Shallow(shape = lout_zy*channel_bb[-1]),
        #             nn.Linear(lout_zy*channel_bb[-1],channel_bb[-1]),
        #             nn.BatchNorm1d(channel_bb[-1]),
        #             nn.LeakyReLU(1.0,inplace=True)
        #         ]
        # self.branch_broadband.append(nn.Linear(lout_zy,channel_bb[-1]))
        # self.branch_broadband.append(nn.BatchNorm1d(channel_bb[-1]))

        # self.branch_common.append(Squeeze())
        # self.branch_common +=[
        #             Shallow(shape = lout_zyx*channel_com[-1]),
        #             nn.Linear(lout_zyx*channel_com[-1],channel_com[-1]),
        #             nn.BatchNorm1d(channel_com[-1]),
        #             nn.LeakyReLU(1.0,inplace=True)
        #         ]
        # self.branch_common.append(nn.Linear(lout_zyx,channel_com[-1]))
        # self.branch_common.append(nn.BatchNorm1d(channel_com[-1]))
        # pdb.set_trace()

        self.master         = nn.Sequential(*self.cnn1)
        self.cnn_common     = nn.Sequential(*self.branch_common)
        self.cnn_broadband  = nn.Sequential(*self.branch_broadband)

        # self.cnn_common = nn.Sequential(*(self.cnn1+self.branch_common))
        # self.cnn_broadband = nn.Sequential(*(self.cnn1 + self.branch_broadband))
        # self.zyx        = nn.Sequential(*self.branch_common)

    def forward(self, x):
        z   = self.master(x)
        zyx = self.cnn_common(z)
        zy  = self.cnn_broadband(z)

        if not self.training:
            zy, zyx = zy.detach(), zyx.detach()
        if self.wf:
            return  zy, zyx 
        return zy, zyx

class Encoder_Unic_Resnet(BasicEncoderDataParallele):
    """docstring for Encoder"""
    def __init__(self, ngpu,dev,nz,nch,ndf,act,channel,\
                 nly, config,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256, path='',dconv = "",\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2, wf = False, *args, **kwargs):
        # pdb.set_trace()
        super(Encoder_Unic_Resnet, self).__init__(*args, **kwargs)
        self.ngpu= ngpu
        self.gang = range(ngpu)
        self.wf = wf

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
        dpc_bb      = config["broadband"]["dpc"]
        extra_bb    = config["broadband"]["extra"]
        acts_bb     = T.activation(config["broadband"]["act"], config["broadband"]["nlayers"])

        #common part
        channel_com  = config["common"]["channel"]
        ker_com      = config["common"]["kernel"]
        std_com      = config["common"]["strides"]
        dil_com      = config["common"]["dilation"]
        pad_com      = config["common"]["padding"]
        nly_com      = config["common"]["nlayers"]
        dpc_com      = config["common"]["dpc"]
        extra_com    = config["common"]["extra"]
        acts_com     = T.activation(config["common"]["act"], config["common"]["nlayers"])

        
        lout = self.lout(nch=4096, 
            padding    =pad, 
            dilation   =dil,
            stride     =std,
            kernel_size=ker)

        lout_zy = self.lout(nch=lout, 
            padding    =config["broadband"]["padding"], 
            dilation   =config["broadband"]["dilation"],
            stride     =config["broadband"]["strides"],
            kernel_size=config["broadband"]["kernel"])

        lout_zyx = self.lout(nch=lout, 
            padding     =config["common"]["padding"], 
            dilation    =config["common"]["dilation"],
            stride      =config["common"]["strides"],
            kernel_size =config["common"]["kernel"])

        for i in range(1, nly+1):
            # ker[i-1] = self.kout(nly,i,ker)
            # lout = self.lout(lin, std, ker, pad, dil)
            # _pad = self.pad(nly,i,pad,lin,lout, dil, ker, std)
            # lin = lout
            if i ==1 and with_noise:
            #We take the first layers for adding noise to the system if the condition is set
                self.cnn1 = self.cnn1+cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil = dil[i-1], dpc=0.0,wn=True ,\
                    dtm = dtm, ffr = ffr, wpc = wpc,dev='cuda:0')
            # esle if we not have a noise but we at the strating network
            elif i == 1:
                self.cnn1 = self.cnn1+cnn1d(channel[i-1],channel[i], acts[0],ker=ker[i-1],std=std[i-1],\
                    pad=pad[i-1],bn=bn,dil=dil[i-1],dpc=0.0,wn=False)
            #else we proceed normaly
            else:
                _bn  = False if i == nly else bn
                _dpc = 0.0 if i == nly else dpc 
                self.cnn1 = self.cnn1+cnn1d(channel[i-1], channel[i], acts[i-1], ker=ker[i-1],\
                    std=std[i-1],pad=pad[i-1], dil=dil[i-1], bn=_bn, dpc=_dpc, wn=False)

        # # pdb.set_trace()
        # for n in range(1,nly_bb+1):
        #     _bn  = False if n==nly_bb else bn
        #     _dpc = 0.0   if n==nly_bb else dpc_bb
        #     self.branch_broadband += cnn1d(channel_bb[n-1],channel_bb[n],\
        #         acts_bb[n-1],ker=ker_bb[n-1],std=std_bb[n-1],\
        #         pad=pad_bb[n-1],bn=_bn,dil=dil_bb[n-1],dpc=_dpc,wn=False)

        # net = EncoderResnet(in_signals_channels = 32, out_signals_channels=8,
        #     channels = [32,64], layers = [2,2], block = block_2x2
        # )

        self.branch_common +=[
            nn.Flatten(start_dim = 1, end_dim=2),
            Linear(lout_zyx*channel_com[-1],128),
            nn.LeakyReLU(1.0,inplace=True)
        ]

        self.branch_broadband +=[
            nn.Flatten(start_dim = 1, end_dim=2),
            Linear(lout_zy*channel_bb[-1],384),
            nn.LeakyReLU(1.0,inplace=True)
        ]

        self.master = nn.Sequential(*self.cnn1)
        self.branch_common      = nn.Sequential(*self.branch_common)
        self.branch_broadband   = nn.Sequential(*self.branch_broadband)

        self.cnn_common         = EncoderResnet(in_signals_channels = 32, out_signals_channels=4,
                                    channels = [32,64], layers = [2,2], block = block_2x2)

        self.cnn_broadband      = EncoderResnet(in_signals_channels = 32, out_signals_channels=4,
                                    channels = [32,64], layers = [2,2], block = block_2x2)

        # self.cnn_common = nn.Sequential(*(self.cnn1+self.branch_common))
        # self.cnn_broadband = nn.Sequential(*(self.cnn1 + self.branch_broadband))
        # self.zyx        = nn.Sequential(*self.branch_common)

    def forward(self, x):
        z   = self.master(x)
        zyx = self.cnn_common(z)
        zy  = self.cnn_broadband(z)
        # breakpoint()
        #Flatten results
        zyx = self.branch_common(zyx)
        zy  = self.branch_broadband(zy)

        if not self.training:
            zy, zyx = zy.detach(), zyx.detach()
        if self.wf:
            return  zy, zyx 
        return zy, zyx


class Encoder_Octave(BasicEncoderDataParallele): 
    """ This class is an encdder class based on the octave convolution.
        With this implementation we try to enforce the calculations
        for High Frequency and Low Frequency signals. 
    """
    def __init__(self,  ngpu,dev,nz,nch,ndf,act,channel,
        nly, config,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,\
        dpc=0.0,limit = 256, path='',dconv = "",\
        with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2,wf = False, *args, **kwargs):
        
        super(Encoder_Octave, self).__init__(*args, **kwargs)
        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")
        self.leaky_relu = nn.LeakyReLU(1.0,inplace=True)

        # channel = config["common"]["channel"]
        self.layers = []
        self.wf = wf

        for i in range(nly): 
             self.layers += [OctaveBatchNormActivation(
                conv           = OctaveConv, 
                in_channels    = channel[i], 
                out_channels   = channel[i+1], 
                stride         = std[i],
                kernel_size    = ker[i], 
                padding        = pad[i], 
                dilation       = dil[i], 
                activation_layer = act
        )]


        self.conv = nn.Sequential(*self.layers)
        self.conv.to(self.device)
        
        self.conv_h  = nn.Sequential(*[
            nn.Conv1d(in_channels = limit[0], out_channels =limit[1], stride=2,
                                 kernel_size=3, padding  = 1),
            self.leaky_relu]).to(self.device)

        self.conv_l  = nn.Sequential(*[
            nn.Conv1d(in_channels = limit[0], out_channels = limit[2], stride=1,
                                 kernel_size=3, padding = 1),
            self.leaky_relu]).to(self.device)

    def forward(self,x):
        # pdb.set_trace()
        z_h, z_l = self.conv(x)
        z_h      = self.conv_h(z_h)
        z_l      = self.conv_l(z_l)
        #empty tensor to avoid stopping the training
        z_0      = torch.empty(0).to(self.device) 
        return z_h, z_l, z_0


class Encoder_ResNet(BasicEncoderDataParallele):
    """docstring for Encoder"""
    def __init__(self, ngpu,dev,nz,nch,ndf,act,channel,\
                 nly, config,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.0,limit = 256, path='',dconv = "",\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2,
                 wf = False, *args, **kwargs):
        super(Encoder_ResNet, self).__init__(*args, **kwargs)
        self.ngpu= ngpu
        self.gang = range(ngpu)
        self.wf = wf
        
        self.device = tdev("cuda" if torch.cuda.is_available() else "cpu")

        acts = T.activation(act, nly)
        # pdb.set_trace()

        self.cnn_common  = ResNetEncoder(in_channels = channel[0], blocks_sizes= channel[1:],
                                         deepths=limit, activation='leaky_relu')

        self.zyx         = ResNetEncoder(in_channels = channel[-1], blocks_sizes=config["common"]["channel"], 
                                        deepths=config["common"]["limit"], activation='leaky_relu')

        self.zy          = ResNetEncoder(in_channels = channel[-1], blocks_sizes=config["broadband"]["channel"],
                                         deepths=config["broadband"]["limit"], activation='leaky_relu')

        self.zx          = ResNetEncoder(in_channels = channel[-1], blocks_sizes=config["filtered"]["channel"],
                                         deepths=config["filtered"]["limit"], activation='leaky_relu')
        # pdb.set_trace()
        self.cnn_common.to(self.device)
        self.zy.to(self.device)
        self.zyx.to(self.device)
        self.zx.to(self.device)

    def forward(self,x):
        # pdb.set_trace()
        if x.is_cuda and self.ngpu >=1:

            z   = pll(self.cnn_common,x,self.gang)
            zy  = pll(self.zy,  z, self.gang)
            zyx = pll(self.zyx, z, self.gang)
            zx  = pll(self.zx, z, self.gang)
            # z   = self.cnn_common(x)
            # zy  = self.zy(z)
            # zyx = self.zyx(z)
            # zx  = self.zx(z)
        else:
            z   = self.cnn_common(x)
            zy  = self.zy(z)
            zyx = self.zyx(z)
            zx  = self.zx(z)
        if not self.training:
            zy, zyx, zx = zy.detach(), zyx.detach(), zx.detach()
        return zy, zyx, zx 


class Encoder_Lite(object):
    """docstring for Encoder_Lite"""
    def __init__(self, arg):
        super(Encoder_Lite, self).__init__()
        pass

    def forward(self, *input): 
        pass
        

