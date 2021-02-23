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

class DCGAN_DxModelParallele(object):
    """docstring for DCGAN_Dx"""
    def __init__(self, *arg, **kwargs):
        super(DCGAN_DxModelParallele, self).__init__()
        pass

    @staticmethod
    def getDCGAN_DxByGPU(ngpu, nc, ncl, ndf, nly,act,channel, fpd=0,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.25, limit = 256,
                 n_extra_layers=0,isize=256):
        classname = 'DCGAN_Dx_' + str(ngpu)+'GPU'
        #this following code is equivalent to calls the class it self. 
        """
        Here we define a methode, in which the purpose is to call and return a class by the name 
        This is made in the purpose of spliting the whole network in the GPUs allowed. There for 
        This suppose that one node of ngpu is present in the configuraton of the environement. 
        """  
        module_name = "dcgan_dxmodelparallele"
        module = importlib.import_module(module_name)
        class_ = getattr(module,classname)

        return class_(ngpu = ngpu, isize = isize, nc = nc, ncl = ncl, ndf = ndf, channel = channel, fpd = fpd, act=act,\
                        nly = nly, ker=ker ,std=std, pad=pad, dil=dil, grp=grp, bn=bn, wf = wf, dpc=dpc,\
                        n_extra_layers = n_extra_layers, limit = limit)


class BasicDCGAN_Dx(Module):
    """docstring for BasicDCGAN_Dx"""
    def __init__(self, *arg,**kwargs):
        super(BasicDCGAN_Dx, self).__init__()
        #ordinal attribute of the GPUs
        self.dev0 = 0
        self.dev1 = 1
        self.dev2 = 2
        self.dev3 = 3

        #initlialization of the GPUs
        self.cnn1 = []
        self.cnn2 = []
        self.cnn3 = []
        self.cnn4 = []

        #trainings
        self.training = True
        self.wf       = True
        self.prc      = []
        self.exf      = []
        self.extra    = []
        self.final    = []

    def lout(self, nz, nly, increment, limit):
        #Here we specify the logic of the  in_channels/channel[i]
        n = nz*2**(increment)
        #we force the last of the channel[i] to not be greater than 512
        val = n if (n<limit or increment<nly) else limit
        return val if val <= limit else limit
    
    def kout(self, nly, incremement, ker):
        return ker if incremement <= nly//2+1 else ker*2

    def pad(self, nly, incremement, pad):
        return 4 if incremement == nly else pad

    def critic(self,X):
        pass

    def extraction(self,X):
        pass


class DCGAN_Dx_1GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_1GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly, act, channel, fpd=1, isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250,
                 n_extra_layers=0):
        super(DCGAN_Dx_1GPU, self).__init__()
        #activation code
        activation = T.activation(act, nly)
        
        #extraction features 
        self.wf = wf

        #building network
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2, nly+1):
            act = activation[i-1]
            _bn = False if i == 1 else bn
            _dpc = 0.0 if i == nly else dpc
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act)) 

        """
            The kernel = 3 and the stride = 1 not change the third dimension
        """
        for _ in range(0,n_extra_layers):
            self.extra+=[Conv1d(in_channels = channel[i],out_channels=channel[i],\
                kernel_size = 3, stride = 1, padding=1, bias=False)]

        self.final+=[Conv1d(channel[i], channel[i], 3, padding=1, bias=False)]
        self.final+=[BatchNorm1d(channel[i])]
        self.final+=[Dpout(dpc=dpc)]
        self.final+=[activation[-1]]

        #compute values 
        self.exf  = self.cnn1
        self.cnn1 = self.prc + self.cnn1 + self.extra + self.final

        #creating sequentially the Network
        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype = torch.float32)

        self.prc = sqn(*self.prc)
        self.prc.to(self.dev0, dtype = torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(channel[i], 1),
            torch.nn.Sigmoid()
        ).to(self.dev0, dtype=torch.float32)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self,x):
        # pdb.set_trace()
        x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(x)
        torch.cuda.empty_cache()
        if self.wf:
            f = self.extraction(x)
        # torch.cuda.empty_cache()
        if not self.training:
            z=z.detach()
        if self.wf:
            return z,f
        else:
            return z

    def critc(self,X):
        X = forward(X)
        return self.features_to_prob(X)



class DCGAN_Dx_2GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_2GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel,fpd=1,isize=256, limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250,
                 n_extra_layers=0):
        super(DCGAN_Dx_2GPU, self).__init__()
        self.ngpu= ngpu


        #extraction features 
        self.wf = wf

        #building network
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))


        #part I in the GPU0
        in_channels   = nc
        for i in range(2, nly//2+1):
            
            """
            The whole value of stride, kernel and padding should be generated accordingn to the 
            pytorch documentation:
            https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            """
            _bn = False if i == 1 else bn
            _dpc = 0.0 if i == nly else dpc
            act = activation[i-1]
            # self.cnn1 += cnn1d(in_channels, channel[i], act,ker=ker,std=std,pad=pad,\
            #     dil =dil,bn=_bn,dpc=_dpc)
            _bn = bn if i == 1 else True
            self.cnn1.append(ConvBlock(ni = channel[i-1], no = channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))

        """
        The second loop generate the second network, which should be saved after in the GPU1
        """
        #Part II in the GPU1
        for i in range(nly//2+1, nly+1):
            
            act = activation[i-1]
            _bn = bn if i == 1 else True
            self.cnn2.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))

        for _ in range(0,n_extra_layers):
            self.extra+=[Conv1d(in_channels = channel[i],out_channels=channel[i],\
                kernel_size = 3, stride = 1, padding=1, bias=False)]

        self.final+=[Conv1d(channel[i], channel[i], 3, padding=1, bias=False)]
        self.final+=[BatchNorm1d(channel[i])]
        self.final+=[Dpout(dpc=dpc)]
        self.final+=[activation[-1]]

        """
        Here we define put the network and the GPUs
        we precise that the type will be in float32
        non_blocking = True to free memory when it is not needed anymore
        """
        self.exf  = self.cnn1 + self.cnn2
        self.cnn1 = self.prc + self.cnn1
        self.cnn2 = self.cnn2 + self.extra + self.final

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.features_to_prob = torch.nn.Sequential(
            torch.nn.Linear(channel[i], 1),
            torch.nn.Sigmoid()
        ).to(self.dev1, dtype=torch.float32)

    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f

    def forward(self, x):
        x = x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(x)
        
        z = z.to(self.dev1,dtype=torch.float32)
        z = self.cnn2(z)
        
        if self.wf:
            f = self.extraction(x)

        if not self.training:
            z=z.detach()

        if self.wf:
            return z,f
        else:
            return z

    def critc(self,X):
        X = forward(X)
        return self.features_to_prob(X)


class DCGAN_Dx_3GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_3GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel, fpd=1, isize=256,limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.250,
                 n_extra_layers=0):
        super(DCGAN_Dx_3GPU, self).__init__()
        
        #extraction features 
        self.wf = wf

        #building network
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        for i in range(2, nly//3+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # _bn = False if i == 1 else bn
            # self.cnn1 += cnn1d(in_channels, channel[i], act,ker=ker,std=std,pad=pad,\
            #     dil=dil, bn=_bn,dpc=dpc)
            _bn = bn if i == 1 else True
            self.cnn1.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))
            
        for i in range(nly//3+1, 2*nly//3+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels, channel[i], act,ker=ker,std=std,pad=pad,\
            #     dil =dil, bn=bn,dpc=dpc)
            _bn = bn if i == 1 else True
            self.cnn2.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))            

        for i in range(2*nly//3+1, nly+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # _dpc = 0.0 if i == nly else dpc
            # self.cnn3 += cnn1d(in_channels,channel[i], act, ker=ker,std=std,pad=pad,dil =dil,\
            #         bn=bn,dpc=_dpc,wn=False)
            _bn = bn if i == 1 else True
            self.cnn3.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))
            

        for _ in range(0,n_extra_layers):
            self.extra+=[Conv1d(in_channels = channel[i],out_channels=channel[i],\
                kernel_size = 3, stride = 1, padding=1, bias=False)]

        self.final+=[Conv1d(channel[i], channel[i], 3, padding=1, bias=False)]
        self.final+=[BatchNorm1d(channel[i])]
        self.final+=[Dpout(dpc=dpc)]
        self.final+=[activation[-1]]

        self.exf  = self.cnn1 + self.cnn2 +  self.cnn3
        self.cnn1 = self.prc + self.cnn1
        self.cnn3 = self.cnn3 + self.extra + self.final

        self.cnn1 = sqn(*self.cnn1)
        self.cnn2 = sqn(*self.cnn2)
        self.cnn3 = sqn(*self.cnn3)

        self.cnn1.to(self.dev0,dtype=torch.float32)
        self.cnn2.to(self.dev1,dtype=torch.float32)
        self.cnn3.to(self.dev2,dtype=torch.float32)
        
    def forward(self, x):
        z = x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(z)
        z = z.to(self.dev1,dtype=torch.float32)
        z = self.cnn2(z)
        z = z.to(self.dev2,dtype=torch.float32)
        z = self.cnn3(z)
        # torch.cuda.empty_cache()
        if not self.training:
            z = z.detach()

        if self.wf:
            return z,f
        else:
            return z

class DCGAN_Dx_4GPU(BasicDCGAN_Dx):
    """docstring for DCGAN_Dx_4GPU"""
    def __init__(self, ngpu, nc, ncl, ndf, nly,act, channel, fpd=1, isize=256,limit = 256,\
                 ker=2,std=2,pad=0, dil=1,grp=1,bn=True,wf=False, dpc=0.25,
                 n_extra_layers=0):
        super(DCGAN_Dx_4GPU, self).__init__()

        #extraction features 
        self.wf = wf

        #building network
        self.prc.append(ConvBlock(ni = channel[0], no = channel[1],
                ks = ker[0], stride = std[0], pad = pad[0], dil = dil[0],\
                bn = False, act = activation[0], dpc = dpc))

        #Part I in GPU0
        for i in range(2, nly//4+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # _bn = False if i == 1 else bn
            # self.cnn1 += cnn1d(in_channels, channel[i], act, ker=ker,std=std,pad=pad,dil =dil,\
            #     bn=_bn,dpc=dpc)
            _bn = bn if i == 1 else True
            self.cnn1.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))
            

        #Part II in GPU1
        for i in range(nly//4+1, 2*nly//4+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # self.cnn2 += cnn1d(in_channels, channel[i], act,ker=ker,std=std,pad=pad,dil =dil,\
            #     bn=bn,dpc=dpc)
            _bn = bn if i == 1 else True
            self.cnn2.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))
            
        #Part III in GPU2
        for i in range(2*nly//4+1, 3*nly//4+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # self.cnn3 += cnn1d(in_channels, channel[i], act,ker=ker,std=std,pad=pad,dil =dil,\
            #     bn=bn,dpc=dpc)
            _bn = bn if i == 1 else True
            self.cnn3.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))
            

        #Part IV in GPU3
        for i in range(3*nly//4+1, nly+1):
            
            # _ker = self.kout(nly,i,ker)
            # _pad = self.pad(nly,i,pad)
            act = activation[i-1]
            # _dpc = 0.0 if i == nly else dpc
            # self.cnn4 += cnn1d(in_channels,channel[i], act,ker=ker,std=std,pad=pad,dil =dil,\
            #         bn=bn,dpc=_dpc,wn=False)
            _bn = bn if i == 1 else True
            self.cnn4.append(ConvBlock(ni = channel[i-1], no =channel[i],
                ks = ker[i-1], stride = std[i-1], pad = pad[i-1], dil = dil[i-1], bias = False,\
                bn = _bn, dpc = dpc, act = act))            
            

        for _ in range(0,n_extra_layers):
            self.extra+=[Conv1d(in_channels = channel[i],out_channels=channel[i],\
                kernel_size = 3, stride = 1, padding=1, bias=False)]

        self.final+=[Conv1d(channel[i], channel[i], 3, padding=1, bias=False)]
        self.final+=[BatchNorm1d(channel[i])]
        self.final+=[Dpout(dpc=dpc)]
        self.final+=[activation[-1]]

        self.exf  = self.cnn1 + self.cnn2 + self.cnn3 + self.cnn4
        self.cnn1 = self.prc + self.cnn1
        self.cnn4 = self.cnn4 + self.extra + self.final

        self.cnn1 = sqn(*self.cnn1)
        self.cnn1.to(self.dev0, dtype=torch.float32)

        self.cnn2 = sqn(*self.cnn2)
        self.cnn2.to(self.dev1, dtype=torch.float32)

        self.cnn3 = sqn(*self.cnn3)
        self.cnn3.to(self.dev2, dtype=torch.float32)

        self.cnn4 = sqn(*self.cnn4)
        self.cnn4.to(self.dev3, dtype=torch.float32)


    def forward(self, x):
        z = x.to(self.dev0,dtype=torch.float32)
        z = self.cnn1(z)
        z = z.to(self.dev1,dtype=torch.float32)
        z = self.cnn2(z)
        z = z.to(self.dev2,dtype=torch.float32)
        z = self.cnn3(z)
        z = z.to(self.dev3,dtype=torch.float32)
        z = self.cnn4(z)
        
        if self.wf:
            f = self.extraction(x)

        if not self.training:
            z = z.detach()
            
        if self.wf:
            return z,f
        else:
            return x