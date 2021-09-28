from torch.nn.modules import activation

u'''AE design'''
from torch.nn.modules import activation
u'''AE design'''
u'''Required modules'''
import warnings
import GPUtil
warnings.filterwarnings("ignore")
from common_nn import *
import torch
import pdb
from torch import device as tdev

class DenseEncoder(Module):
    def __init__(self,ngpu,dev,ninp,nout,szs,nly,
                 act=[LeakyReLU(1.0,True),Tanh()],dpc=0.10):
        super(DenseEncoder,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev = dev
        self.cnn = []
        for l in range(nly):
            self.cnn.append(ConvBlock(ni=ninp[l],no=nout[l],ks=szs-1,
                                      stride=1,bias=True,act=act[l],
                                      bn=False))
        self.cnn = sqn(*self.cnn)
        
    def forward(self,Xn):
        if Xn.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,Xn,self.gang)
        else:
            zlf   = self.cnn(Xn)
        if not self.training:
            zlf=zlf.detach()
        return zlf
    
class ResEncoder(Module):
    def __init__(self,ngpu,dev,nz,nzcl,nch,ndf,\
                 szs,nly,ker=7,std=4,pad=0,dil=1,grp=1,\
                 dpc=0.10,act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),Tanh()],\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=2.e-2):
        super(ResEncoder,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev = dev
        
        self.cnn = [ConvBlock(ni=nch*1,no=ndf*2,ks=ker,stride=std,act=act[0],bn=False),
                    ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[1],bn=True ,dpc=dpc),
                    ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[2],bn=True ,dpc=dpc),
                    ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[3],bn=True ,dpc=dpc),
                    ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[4],bn=True,dpc=dpc)]
        
        self.cnn = [ResNet(ann=self.cnn),ConvBlock(ni=ndf*2,no=nz,ks=33,pad=16,stride=16,act=act[-1],bn=False)]
        
        self.cnn = sqn(*self.cnn)
        
    def forward(self,Xn):
        if Xn.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,Xn,self.gang)
        else:
            zlf   = self.cnn(Xn)
        if not self.training:
            zlf=zlf.detach()
        return zlf

class EncoderDecoder(Module):
    def __init__(self,ngpu,dev,nz,nch,ndf,nly,ker=7,std=4,opd=0,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.10,act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),
                               LeakyReLU(1.0,True),LeakyReLU(1.0,True),
                               LeakyReLU(1.0,True),Tanh()]):
        super(EncoderDecoder,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev = dev
        if nly==3:
            self.cnn = \
                cnn1d(nch*2,ndf*1,act[0],ker=ker,std=std,pad=pad+1,bn=bn,dpc=dpc  ,wn=False)+\
                cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad+1,bn=bn,dpc=dpc  ,wn=False)+\
                cnn1d(ndf*2,nz ,act[2],ker=ker,std=std,pad=pad+1,bn=False,dpc=0.0 ,wn=True,dev=self.dev)+\
                cnn1dt(2*nz ,ndf*2,act[3],ker=ker,std=std,pad=1,opd=opd,bn=True, dpc=dpc)+\
                cnn1dt(ndf*2,ndf*1,act[4],ker=ker,std=std,pad=1,opd=opd,bn=True, dpc=dpc)+\
                cnn1dt(ndf*1,nch*1,act[5],ker=ker,std=std,pad=1,opd=opd,bn=False, dpc=0.0)
        elif nly==5:
            self.cnn = \
                cnn1d(nch*2,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0   ,wn=False)+\
                cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc   ,wn=False)+\
                cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc   ,wn=False)+\
                cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc   ,wn=False)+\
                cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0,wn=True)+\
                cnn1dt(2*nz,ndf*8,act[5],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                cnn1dt(ndf*8,ndf*4,act[6],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                cnn1dt(ndf*4,ndf*2,act[7],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                cnn1dt(ndf*2,ndf*1,act[8],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                cnn1dt(ndf*1,nch*1,act[9],ker=ker,std=std,pad=1,opd=opd,bn=False, dpc=0.0)
        self.cnn = sqn(*self.cnn)
        
        
    def forward(self,Xn):
        if Xn.is_cuda and self.ngpu > 1:
            zlf   = pll(self.cnn,Xn,self.gang)
        else:
            zlf   = self.cnn(Xn)
        if not self.training:
            zlf=zlf.detach()
        return zlf

class Encoder(Module):
    def __init__(self,ngpu,dev,nz,nzcl,nch,ndf,\
                 szs,nly,ker=7,std=4,pad=0,dil=1,grp=1,bn=True,
                 dpc=0.10,act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),Tanh()],\
                 with_noise=False,dtm=0.01,ffr=0.16,wpc=5.e-2):
        super(Encoder,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev = dev
        # pdb.set_trace()
        if ngpu ==1:
            if nly==3:
                # 3 layers
                if with_noise:
                    self.cnn = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=False,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)+\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn ,dpc=dpc,wn=False)+\
                        cnn1d(ndf*2,nz   ,act[2],ker=ker,std=std,pad=pad,bn=False,dpc=0.0,wn=False)
                else:
                    self.cnn = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*2,nz ,act[2],ker=ker,std=std,pad=pad,bn=False,dpc=0.0,wn=False)
            elif nly==5:
                # 5 layers
                if with_noise:
                    self.cnn = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)+\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0,wn=False)
                else:
                    self.cnn = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0)+\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=pad  ,bn=False,dpc=0.0)
            elif nly==6:
                # 6 layers
                if with_noise:
                    self.cnn = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)+\
                        cnn1d(ndf*1 ,ndf*2 ,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*2 ,ndf*4 ,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*4 ,ndf*8 ,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*8 ,ndf*16,act[4],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*16,nz    ,act[5],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0,wn=False)
                else:
                    self.cnn = \
                        cnn1d(nch*1 ,ndf*1 ,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0)+\
                        cnn1d(ndf*1 ,ndf*2 ,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*2 ,ndf*4 ,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*4 ,ndf*8 ,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*8 ,ndf*16,act[4],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*16,nz    ,act[5],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0)
            elif nly==8:
                self.cnn = \
                    cnn1d(nch    ,ndf*1  ,act[0],ker=ker,std=std,pad=pad,bn=bn , dpc=0.0)+\
                    cnn1d(ndf*1  ,ndf*2  ,act[1],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*2  ,ndf*4  ,act[2],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*4  ,ndf*8  ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*8  ,ndf*16 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*16 ,ndf*32 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*32 ,ndf*64 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*64 ,nz,act[4],ker=ker,std=std,pad=pad,bn=False, dpc=0.0)


            self.cnn = sqn(*self.cnn)
            self.cnn.to(self.dev,non_blocking=True,dtype=torch.float32)
        else:
            if ngpu==2:
                self.dev0 = 0
                self.dev1 = 0
                self.dev2 = 1
                self.dev3 = 1
            elif ngpu == 3:
                self.dev0 = 0
                self.dev1 = 1
                self.dev2 = 2
                self.dev3 = 2
            elif ngpu == 4:
                self.dev0 = 0
                self.dev1 = 1
                self.dev2 = 2
                self.dev3 = 3
            if nly==5:
                # 5 layers
                if with_noise:
                    self.cnn1=\
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)
                    self.cnn2=\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn3=\
                        cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn4=\
                        cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=pad  ,bn=False,dpc=0.0,wn=False)
                else:
                    self.cnn1 =\
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0)
                    self.cnn2 =\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)
                    self.cnn3 =\
                        cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)
                    self.cnn4 =\
                        cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=pad  ,bn=False,dpc=0.0)
            elif nly==6:
                # 6 layers
                if with_noise:
                    self.cnn1 =\
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=True,dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)+\
                        cnn1d(ndf*1 ,ndf*2 ,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn2 =\
                        cnn1d(ndf*2 ,ndf*4 ,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*4 ,ndf*8 ,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn3 =\
                        cnn1d(ndf*8 ,ndf*16,act[4],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn4 =\
                        cnn1d(ndf*16,nz    ,act[5],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0,wn=False)
                else:
                    self.cnn1 =\
                        cnn1d(nch*1 ,ndf*1 ,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0)+\
                        cnn1d(ndf*1 ,ndf*2 ,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)
                    self.cnn2 =\
                        cnn1d(ndf*2 ,ndf*4 ,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*4 ,ndf*8 ,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)
                    self.cnn3 =\
                        cnn1d(ndf*8 ,ndf*16,act[4],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)
                    self.cnn4 =\
                        cnn1d(ndf*16,nz    ,act[5],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0)
            elif nly==8:
                self.cnn1=\
                    cnn1d(nch    ,ndf*1  ,act[0],ker=ker,std=std,pad=pad,bn=bn , dpc=0.0)+\
                    cnn1d(ndf*1  ,ndf*2  ,act[1],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)
                self.cnn2=\
                    cnn1d(ndf*2  ,ndf*4  ,act[2],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*4  ,ndf*8  ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)
                self.cnn3=\
                    cnn1d(ndf*8  ,ndf*16 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*16 ,ndf*32 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)
                self.cnn4=\
                    cnn1d(ndf*32 ,ndf*64 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*64 ,nz,act[4],ker=ker,std=std,pad=pad,bn=False, dpc=0.0)

            self.cnn1 = sqn(*self.cnn1)
            self.cnn2 = sqn(*self.cnn2)
            self.cnn3 = sqn(*self.cnn3)
            self.cnn4 = sqn(*self.cnn4)
            self.cnn1.to(self.dev0,non_blocking=True,dtype=torch.float32)
            self.cnn2.to(self.dev1,non_blocking=True,dtype=torch.float32)
            self.cnn3.to(self.dev2,non_blocking=True,dtype=torch.float32)
            self.cnn4.to(self.dev3,non_blocking=True,dtype=torch.float32)
        
    def forward(self,x):
        if x.is_cuda:
            if self.ngpu > 1:
                #zlf   = pll(self.cnn,Xn,self.gang)
                x = x.to(self.dev0,non_blocking=True,dtype=torch.float32)
                x = self.cnn1(x)
                x = x.to(self.dev1,non_blocking=True,dtype=torch.float32)
                x = self.cnn2(x)
                x = x.to(self.dev2,non_blocking=True,dtype=torch.float32)
                x = self.cnn3(x)
                x = x.to(self.dev3,non_blocking=True,dtype=torch.float32)
                z = self.cnn4(x)
            elif self.ngpu == 1:
                x = x.to(self.dev,non_blocking=True,dtype=torch.float32)
                z = self.cnn(x)
            torch.cuda.empty_cache()
        else:
            z = self.cnn(x.to(torch.float32))
        if not self.training:
            z = z.detach()
        return z

class Decoder(Module):
    def __init__(self,ngpu,dev,nz,nch,ndf,nly,\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,\
                 act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),\
                      LeakyReLU(1.0,True),LeakyReLU(1.0,True)]):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev = dev
        if ngpu ==1:
            if nly==3:
                self.cnn = \
                    cnn1dt(nz   ,ndf*8,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8,ndf*4,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4,nch  ,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            elif nly==4:
                self.cnn = \
                    cnn1dt(nz   ,ndf*8,act[0],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8,ndf*4,act[1],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4,ndf*2,act[2],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*2,ndf*1,act[3],ker=ker,std=std,pad=1,opd=opd,bn=False, dpc=0.0)
            elif nly==5:
                self.cnn = \
                    cnn1dt(nz   ,ndf*8,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8,ndf*4,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4,ndf*2,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*2,ndf*1,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*1,nch*1,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            elif nly==8:
                self.cnn = \
                    cnn1dt(nz    ,ndf*64,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*64,ndf*32,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*32,ndf*16,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*16,ndf*8 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8 ,ndf*4 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4 ,ndf*2 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*2 ,ndf*1 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*1 ,nch*1 ,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            self.cnn = sqn(*self.cnn)
            self.cnn.to(self.dev,non_blocking=True,dtype=torch.float32)
        else:
            if ngpu==2:
                self.dev0 = 0
                self.dev1 = 0
                self.dev2 = 1
                self.dev3 = 1
            elif ngpu == 3:
                self.dev0 = 0
                self.dev1 = 1
                self.dev2 = 2
                self.dev3 = 2
            elif ngpu == 4:
                self.dev0 = 0
                self.dev1 = 1
                self.dev2 = 2
                self.dev3 = 3

            if nly==5:
                self.cnn1 =\
                    cnn1dt(nz   ,ndf*8,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8,ndf*4,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn2 =\
                    cnn1dt(ndf*4,ndf*2,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn3 =\
                    cnn1dt(ndf*2,ndf*1,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn4 =\
                    cnn1dt(ndf*1,nch*1,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            elif nly==6:
                self.cnn1 =\
                    cnn1dt(nz   ,ndf*32,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*32,ndf*16,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn2 =\
                    cnn1dt(ndf*16,ndf*8,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8,ndf*4,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn3 =\
                    cnn1dt(ndf*4,ndf*2,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn4 =\
                    cnn1dt(ndf*2,nch*1,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            elif nly==8:
                self.cnn1 =\
                    cnn1dt(nz    ,ndf*64,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*64,ndf*32,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn2=\
                    cnn1dt(ndf*32,ndf*16,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*16,ndf*8 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn3=\
                    cnn1dt(ndf*8 ,ndf*4 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4 ,ndf*2 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn4=\
                    cnn1dt(ndf*2 ,ndf*1 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*1 ,nch*1 ,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            self.cnn1 = sqn(*self.cnn1)
            self.cnn2 = sqn(*self.cnn2)
            self.cnn3 = sqn(*self.cnn3)
            self.cnn4 = sqn(*self.cnn4)
            self.cnn1.to(self.dev0,non_blocking=True,dtype=torch.float32)
            self.cnn2.to(self.dev1,non_blocking=True,dtype=torch.float32)
            self.cnn3.to(self.dev2,non_blocking=True,dtype=torch.float32)
            self.cnn4.to(self.dev3,non_blocking=True,dtype=torch.float32)


    def forward(self,x):
        if x.is_cuda:
            if self.ngpu > 1:
                # Xr = pll(self.cnn,zxn,self.gang)
                #pdb.set_trace()
                x = x.to(self.dev0,non_blocking=True,dtype=torch.float32)
                x = self.cnn1(x)
                x = x.to(self.dev1,non_blocking=True,dtype=torch.float32)
                x = self.cnn2(x)
                x = x.to(self.dev2,non_blocking=True,dtype=torch.float32)
                x = self.cnn3(x)
                x = x.to(self.dev3,non_blocking=True,dtype=torch.float32)
                Xr = self.cnn4(x)
            elif self.ngpu==1:
                x = x.to(self.dev,non_blocking=True,dtype=torch.float32)
                Xr = self.cnn(x)
            torch.cuda.empty_cache()
        else:
            Xr = self.cnn(x)
        if not self.training:
            Xr = Xr.detach()
        return Xr