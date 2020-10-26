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
        print("Constructor of Encoder...")
        super(Encoder,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev = dev

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
                    cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=pad  ,bn=False,dpc=0.0,wn=False)

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

        self.cnn = sqn(*self.cnn)
        
    def forward(self,Xn):
        if Xn.is_cuda and self.ngpu > 1:
            # splits data in different GPUs if exist
            zlf   = pll(self.cnn,Xn,self.gang)
        else:
            zlf   = self.cnn(Xn)
        if not self.training:
            zlf=zlf.detach()
        return zlf

class Decoder(Module):
    def __init__(self,ngpu,nz,nch,ndf,nly,\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,\
                 act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),\
                      LeakyReLU(1.0,True),LeakyReLU(1.0,True)]):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
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
        self.cnn = sqn(*self.cnn)
            
    def forward(self,zxn):
        if zxn.is_cuda and self.ngpu > 1:
            Xr = pll(self.cnn,zxn,self.gang)
        else:
            Xr = self.cnn(zxn)
        if not self.training:
            Xr=Xr.detach()
        return Xr

class DCGAN_D(Module):
    def __init__(self, ngpu, isize, nc, ncl, ndf, n_extra_layers=0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()], opt=None):
        super(DCGAN_D,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)

        if opt is None :
            initial = [ConvBlock(nc, ndf, 4, 2, bn=False, act=activation[0])]
            
            csize,cndf = isize/2,ndf
            
            extra = [ConvBlock(cndf, cndf, 3, 1, act=activation[0]) 
                     for t in range(n_extra_layers)]

            pyramid = []
            while csize > 4:
                pyramid.append(ConvBlock(cndf, cndf*2, 4, 2,act=activation[0]))
                cndf *= 2; csize /= 2
            final = [Conv1d(cndf, ncl, 4, padding=0, bias=False)]
            print('!!! warnings no configuration found for the DCGAN_D')
        else:
            initial = [ConvBlock(nc, ndf, opt['initial']['kernel'],\
             opt['initial']['strides'],\
             pad=opt['initial']['padding'], bn=False, act=activation[0])]
            
            csize,cndf = isize/2,ndf
            
            extra = [ConvBlock(cndf, cndf, opt['extra']['kernel'],\
             opt['extra']['strides'],pad=opt['extra']['padding'],\
              act=activation[0]) 
                     for t in range(opt['nlayers'])]

            pyramid = []
            while csize > 4:
                pyramid.append(ConvBlock(cndf, cndf*2, opt['pyramid']['kernel'],\
                 opt['pyramid']['strides'],\
                 pad=opt['pyramid']['padding'],act=activation[0]))
                cndf *= 2; csize /= 2
            final = [Conv1d(cndf, ncl, opt['final']['kernel'], padding=opt['final']['padding'], bias=False)]

        ann = initial+extra+pyramid+final+[activation[1]]+[Squeeze()]+[Squeeze()]
        self.ann = sqn(*ann)
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
        else:
            z = self.ann(X)
        if not self.training:
            z = z.detach()
        return z

class DCGAN_Dx(Module):
    def __init__(self, ngpu, isize, nc, ncl, ndf, fpd=0, n_extra_layers=0, dpc=0.0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()],bn=True,wf=False, opt=None):
        super(DCGAN_Dx,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        if opt is None:
            initial = [ConvBlock(nc, ndf, 4, 2, bn=False, act=activation[0],dpc=dpc)]
            csize,cndf = isize/2,ndf
            extra = [ConvBlock(cndf, cndf, 3, 1, bn=bn, act=activation[0],dpc=dpc) 
                     for t in range(n_extra_layers)]

            pyramid = []
            while csize > 16:
                pyramid.append(ConvBlock(cndf, cndf*2, 4, 2, bn=bn, act=activation[0],dpc=dpc))
                cndf *= 2; csize /= 2
            pyramid = pyramid+[ConvBlock(cndf, cndf  , 4, 2, bn=bn, act=activation[0],dpc=dpc)]
            final = [Conv1d(cndf, ncl, 3, padding=fpd, bias=False)]
            if bn: final = final + [BatchNorm1d(ncl)] 
            final = final + [Dpout(dpc=dpc)] + [activation[1]]
        else:
            initial = [ConvBlock(nc, ndf, opt['initial']['kernel'],\
             opt['initial']['stride'],\
             opt['initial']['padding'], bn=False, act=activation[0],dpc=dpc)]
            csize,cndf = isize/2,ndf
            extra = [ConvBlock(cndf, cndf, opt['extra']['kernel'],\
             opt['extra']['strides'],\
             pad=opt['extra']['padding'], bn=bn, act=activation[0],dpc=dpc) 
                     for t in range(n_extra_layers)]
            pyramid = []
            while csize > 16:
                pyramid.append(ConvBlock(cndf, cndf*2,\
                opt['pyramid']['kernel'],\
                opt['pyramid']['strides'], bn=bn, act=activation[0],dpc=dpc))
                cndf *= 2; csize /= 2
            pyramid = pyramid+[ConvBlock(cndf, cndf  ,\
                opt['pyramid']['kernel'],\
                opt['pyramid']['strides'], bn=bn, act=activation[0],dpc=dpc)]
            final = [Conv1d(cndf, ncl, opt['final']['final'],\
             padding=opt['final']['padding'], bias=False)]
            if bn: final = final + [BatchNorm1d(ncl)] 
            final = final + [Dpout(dpc=dpc)] + [activation[1]]

        self.prc = sqn(*initial)
        self.exf = extra+pyramid
        #self.exf = sqn()
        #for i,l in zip(range(extra+pyramid),extra+pyramid):
        #    self.exf.add_module('exf_{}'.format(i),Feature_extractor(l))
                
        ann = initial+extra+pyramid+final#+[Squeeze()]+[Squeeze()]
        self.ann = sqn(*ann)
    
    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.ann(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z
    
class DCGAN_Dz(Module):
    """ create a conv1D layers for the discriminator to tranfrom from (*,*,nz) to (*,*,ncl)
    """
    def __init__(self, ngpu, nz, ncl, fpd=0, n_extra_layers=1, dpc=0.0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()],bn=True,wf=False, opt=None):
        super(DCGAN_Dz,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf

        if opt is None:
            initial =[ConvBlock(nz, ncl, 1, 1, bn=False, act=activation[0],dpc=dpc)]
            layers = [ConvBlock(ncl,ncl, 1, 1, bn=bn, act=activation[0],dpc=dpc) 
                      for t in range(n_extra_layers)]
            
            if bn: layers = layers + [BatchNorm1d(ncl)] 
           
            ann = initial+layers+[Dpout(dpc=dpc)]+[activation[1]] 
            print('warnings no configuration found for the DCGAN_Dz object')
        else:
            initial =[ConvBlock(nz, ncl, opt['initial']['kernel'],\
             opt['initial']['strides'],\
              bn=False, act=activation[0],dpc=dpc)]
            layers = [ConvBlock(ncl,ncl, opt['extra']['kernel'],\
             opt['extra']['strides'], bn=bn, act=activation[0],dpc=dpc) 
                      for t in range(opt['extra']['nlayers'])]
            
            if bn: layers = layers + [BatchNorm1d(ncl)] 
           
            ann = initial+layers+[Dpout(dpc=dpc)]+[activation[1]] 
       
        self.ann = sqn(*ann)
        self.prc = sqn(*initial)
        self.exf = layers[:-1]
    
    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.ann(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        
        if self.wf:
            return z,f
        else:
            return z
    
class DCGAN_G(Module):
    def __init__(self, ngpu, isize, nz, nc, ngf, n_extra_layers=0,
                 activation = [ReLU(inplace=True),Tanh()],dpc=0.0, opt=None):
        super(DCGAN_G,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        cngf, tisize = ngf//2, 4
        if opt is None:
            while tisize!=isize: cngf*=2; tisize*=2
            layers = [DeconvBlock(nz, cngf, 4, 1, 2, act=activation[0],dpc=dpc)]
            
            csize, cndf = 4, cngf
            layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 0, 0, act=activation[0],dpc=dpc))
            
            cngf //= 2; csize *= 2
            while csize < isize//2:
                layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 1,dpc=dpc))
                cngf //= 2; csize *= 2
            
            layers += [DeconvBlock(cngf, cngf, 3, 1, 1, act=activation[0],dpc=dpc) for t in range(n_extra_layers)]
            layers.append(ConvTranspose1d(cngf, nc, 4, 2, 1, bias=False))
            layers.append(activation[1])
            layers.append(Dpout(dpc=dpc))
        else:
            while tisize!=isize: cngf*=2; tisize*=2
            layers = [DeconvBlock(nz, cngf, opt['initial']['kernel'],\
                opt['initial']['strides'],\
                opt['initial']['padding'], act=activation[0],dpc=dpc)]
            
            csize, cndf = 4, cngf
            layers.append(DeconvBlock(cngf, cngf//2, opt['extra']['kernel'],\
                opt['extra']['strides'],\
                opt['extra']['padding'], 0, act=activation[0],dpc=dpc))
            
            cngf //= 2; csize *= 2
            while csize < isize//2:
                layers.append(DeconvBlock(cngf, cngf//2,opt['pyramid']['kernel'],\
                opt['pyramid']['strides'],\
                opt['pyramid']['padding'],dpc=dpc))
                cngf //= 2; csize *= 2
            
            layers += [DeconvBlock(cngf, cngf,opt['before_end']['kernel'],\
                opt['before_end']['strides'],\
                opt['before_end']['padding'], act=activation[0],dpc=dpc) for t in range(opt['nlayers'])]

            layers.append(ConvTranspose1d(cngf, nc, opt['final']['kernel'],\
                opt['final']['strides'],\
                opt['final']['padding'], bias=False))
            layers.append(activation[1])
            layers.append(Dpout(dpc=dpc))

        self.ann = sqn(*layers)
            
    def forward(self,z):
        if z.is_cuda and self.ngpu > 1:
            X = pll(self.ann,z,self.gang)
        else:
            X = self.ann(z)
        if not self.training:
            X = X.detach()
        return X

class DCGAN_DXZ(Module):
    def __init__(self, ngpu, nc, n_extra_layers=0, 
                 activation=[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 dpc=0.0,wf=False, opt=None):
        super(DCGAN_DXZ,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        if opt is None:
            layers = [ConvBlock(nc, nc, 1, 1, bias=True, bn=False, act=activation[0], dpc=dpc) for t in range(n_extra_layers)]
            final  = [ConvBlock(nc,  1, 1, 1, bias=True, bn=False, act=activation[1], dpc=dpc)]
            ann = layers+final
            self.exf = layers
        else:
            layers = [ConvBlock(nc, nc, opt['initial']['kernel'],\
            opt['initial']['strides'],\
            bias=True, bn=False, act=activation[0], dpc=dpc) for t in range(opt['nlayers'])]

            final  = [ConvBlock(nc,  1, opt['final']['kernel'],\
            opt['final']['strides'],\
            bias=True, bn=False, act=activation[1], dpc=dpc)]
            ann = layers+final
            self.exf = layers
        self.ann = sqn(*ann)
        
    def extraction(self,X):
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.ann(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z

class DCGAN_DXX(Module):
    def __init__(self, ngpu, nc, n_extra_layers=0,
                 activation=LeakyReLU(1.0,inplace=True),dpc=0.0, opt=None):
        super(DCGAN_DXX,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        if opt is None:
            layers  = [ConvBlock(nc, nc, 1, 1, bias=False, bn=True, act=activation, dpc=dpc) for t in range(n_extra_layers)]
            final   = [ConvBlock(nc,  1, 1, 1, bias=False, bn=True, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]
        else:
            layers  = [ConvBlock(nc, nc, opt['initial']['kernel'],\
            opt['initial']['strides'],\
            bias=False, bn=True, act=activation, dpc=dpc) for t in range(opt['nlayers'])]
            final   = [ConvBlock(nc,  1, opt['final']['kernel'],\
            opt['final']['strides'], bias=False, bn=True, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]

        self.ann = sqn(*ann)
     
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
        else:
            z = self.ann(X)
        if not self.training:
            z = z.detach()
        return z

class DCGAN_DZZ(Module):
    def __init__(self, ngpu, nc, n_extra_layers=0,
                 activation=LeakyReLU(1.0,inplace=True),dpc=0.0, opt=None):
        super(DCGAN_DZZ,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        if opt is None:
            layers = [ConvBlock(nc, nc, 1, 1, bias=False, bn=False, act=activation, dpc=dpc) for t in range(n_extra_layers)]
            final  = [ConvBlock(nc,  1, 1, 1, bias=False, bn=False, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]#+[Sigmoid()]
        else:
            layers = [ConvBlock(nc, nc, opt['initial']['kernel'],\
             opt['initial']['strides'],\
              bias=False, bn=False, act=activation, dpc=dpc) for t in range(opt['nlayers'])]
            final  = [ConvBlock(nc,  1, opt['final']['kernel'],\
            opt['final']['strides'], bias=False, bn=False, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]#+[Sigmoid()]

        self.ann = sqn(*ann)
     
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
        else:
            z = self.ann(X)
        if not self.training:
            z = z.detach()
        return z
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
from common_nn import *
import torch
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
        else:
            self.dev0 = 0
            self.dev1 = 1
            if nly==5:
                # 5 layers
                if with_noise:
                    self.cnn1 = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)+\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn2=\
                        cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0,wn=False)
                else:
                    self.cnn1 = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0)+\
                        cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)
                    self.cnn2=\
                        cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc)+\
                        cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=pad  ,bn=False,dpc=0.0)
            elif nly==6:
                # 6 layers
                if with_noise:
                    self.cnn1 = \
                        cnn1d(nch*1,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0,wn=True ,\
                              dtm=dtm,ffr=ffr,wpc=wpc,dev=self.dev)+\
                        cnn1d(ndf*1 ,ndf*2 ,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)+\
                        cnn1d(ndf*2 ,ndf*4 ,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc,wn=False)
                    self.cnn2 = \
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
                self.cnn1 = \
                    cnn1d(nch    ,ndf*1  ,act[0],ker=ker,std=std,pad=pad,bn=bn , dpc=0.0)+\
                    cnn1d(ndf*1  ,ndf*2  ,act[1],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*2  ,ndf*4  ,act[2],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*4  ,ndf*8  ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)
                self.cnn2 = \
                    cnn1d(ndf*8  ,ndf*16 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*16 ,ndf*32 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*32 ,ndf*64 ,act[3],ker=ker,std=std,pad=pad,bn=bn , dpc=dpc)+\
                    cnn1d(ndf*64 ,nz,act[4],ker=ker,std=std,pad=pad,bn=False, dpc=0.0)

            self.cnn1 = sqn(*self.cnn1)
            self.cnn2 = sqn(*self.cnn2)
            self.cnn1.to(self.dev0)
            self.cnn2.to(self.dev1)
        
    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            #zlf   = pll(self.cnn,Xn,self.gang)
            x = x.to(self.dev0)
            x = self.cnn1(x)
            x = x.cuda(1)
            x = self.cnn2(x)
            zlf = x
        else:
            zlf   = self.cnn(Xn)
        if not self.training:
            zlf=zlf.detach()
        return zlf

class Decoder(Module):
    def __init__(self,ngpu,nz,nch,ndf,nly,\
                 ker=7,std=4,pad=0,opd=0,dil=1,grp=1,dpc=0.10,\
                 act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),\
                      LeakyReLU(1.0,True),LeakyReLU(1.0,True)]):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
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
                    cnn1dt(nz    ,ndf*128,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*128,ndf*64,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*64,ndf*32,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*32,ndf*16 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*16 ,ndf*8 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8 ,ndf*4 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4 ,ndf*2 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*2 ,nch*1 ,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            self.cnn = sqn(*self.cnn)
        else:
            self.dev0 = 0
            self.dev1 = 1
            if nly==5:
                self.cnn1 = \
                    cnn1dt(nz   ,ndf*8,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8,ndf*4,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn2 = \
                    cnn1dt(ndf*4,ndf*2,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*2,ndf*1,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*1,nch*1,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            elif nly==8:
                self.cnn1 = \
                    cnn1dt(nz    ,ndf*128,act[0],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*128,ndf*64,act[1],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*64,ndf*32,act[2],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*32,ndf*16 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)
                self.cnn2 = \
                    cnn1dt(ndf*16 ,ndf*8 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*8 ,ndf*4 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*4 ,ndf*2 ,act[3],ker=ker,std=std,pad=pad,opd=opd,bn=True , dpc=dpc)+\
                    cnn1dt(ndf*2 ,nch*1 ,act[4],ker=ker,std=std,pad=pad,opd=opd,bn=False, dpc=0.0)
            self.cnn1 = sqn(*self.cnn1)
            self.cnn2 = sqn(*self.cnn2)
            self.cnn1.to(self.dev0)
            self.cnn2.to(self.dev1)


    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            # Xr = pll(self.cnn,zxn,self.gang)
            x = x.to(self.dev0)
            x = self.cnn1(x)
            x = x.cuda(1)
            x = self.cnn2(x)
            Xr = x
        else:
            Xr = self.cnn(zxn)
        if not self.training:
            Xr=Xr.detach()
        return Xr

class DCGAN_D(Module):
    def __init__(self, ngpu, isize, nc, ncl, ndf, n_extra_layers=0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()], opt=None):
        super(DCGAN_D,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.dev0 = 0
        self.dev1 = 1
        if opt is None:
            initial = [ConvBlock(self.ngpu,nc, ndf, 4, 2, bn=False, act=activation[0])]
            
            csize,cndf = isize/2,ndf
            
            extra = [ConvBlock(self.ngpu,cndf, cndf, 3, 1, act=activation[0]) 
                     for t in range(n_extra_layers)]

            pyramid = []
            while csize > 4:
                pyramid.append(ConvBlock(self.ngpu,cndf, cndf*2, 4, 2,act=activation[0]))
                cndf *= 2; csize /= 2
            
            final = [Conv1d(cndf, ncl, 4, padding=0, bias=False)]
            #ann = initial+extra+pyramid+final+[activation[1]]+[Squeeze()]+[Squeeze()]
        else:
            initial = [ConvBlock(self.ngpu,nc, ndf, opt['initial']['kernel'],\
             opt['initial']['strides'], bn=False, act=activation[0])]
            
            csize,cndf = isize/2,ndf
            
            extra = [ConvBlock(self.ngpu,cndf, cndf, opt['extra']['kernel'],\
             opt['extra']['strides'], act=activation[0]) 
                     for t in range(n_extra_layers)]

            pyramid = []
            while csize > 4:
                pyramid.append(ConvBlock(self.ngpu,cndf, cndf*2, opt['pyramid']['kernel'],\
                 opt['pyramid']['strides'],act=activation[0]))
                cndf *= 2; csize /= 2
            
            final = [Conv1d(cndf, ncl, opt['final']['kernel'],\
                opt['final']['strides'],\
                padding=opt['final']['padding'], bias=False)]
        if ngpu==1:
            ann = initial+extra+pyramid+final+[activation[1]]+[Squeeze()]+[Squeeze()]
            self.ann = sqn(*ann)
        else:
            self.ann1 = initial+extra
            self.ann2 = pyramid+final+[activation[1]]+[Squeeze()]+[Squeeze()]
            self.ann1 = sqn(*self.ann1)
            self.ann2 = sqn(*self.ann2)


            self.ann1.to(self.dev0)
            self.ann2.to(self.dev1)

    def forward(self,x):
        if x.is_cuda and self.ngpu > 1:
            #z = pll(self.ann,X,self.gang)
            x = x.to(self.dev0)
            x = self.ann1(x)
            x = x.to(self.dev1)
            x = self.ann2(x)
            z = x
        else:
            z = self.ann(X)
        if not self.training:
            z = z.detach()
        return z

class DCGAN_Dx(Module):
    def __init__(self, ngpu, isize, nc, ncl, ndf, fpd=0, n_extra_layers=0, dpc=0.0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()],bn=True,wf=False, opt=None):
        super(DCGAN_Dx,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        self.dev0 = 0
        self.dev1 = 1

        if opt is None:
            initial = [ConvBlock(self.ngpu,nc, ndf, 4, 2, bn=False, act=activation[0],dpc=dpc)]
            csize,cndf = isize/2,ndf
            extra = [ConvBlock(self.ngpu,cndf, cndf, 4, 4, bn=bn, act=activation[0],dpc=dpc) 
                     for t in range(n_extra_layers)]

            pyramid = []
            while csize > 16:
                pyramid.append(ConvBlock(self.ngpu,cndf, cndf*2, 4, 4, bn=bn, act=activation[0],dpc=dpc))
                cndf *= 2; csize /= 2
            pyramid = pyramid+[ConvBlock(self.ngpu,cndf, cndf  , 4, 2, bn=bn, act=activation[0],dpc=dpc)]
            final = [Conv1d(cndf, ncl, 3, padding=fpd, bias=False)]
            if bn: final = final + [BatchNorm1d(ncl)] 
            final = final + [Dpout(dpc=dpc)] + [activation[1]]
            
            self.prc = sqn(*initial)
            self.exf = extra+pyramid

            #self.exf = sqn()
            #for i,l in zip(range(extra+pyramid),extra+pyramid):
            #    self.exf.add_module('exf_{}'.format(i),Feature_extractor(l))
            #ann = initial+extra+pyramid+final#+[Squeeze()]+[Squeeze()]
            #self.ann = sqn(*ann)
        else:
            initial = [ConvBlock(self.ngpu,nc, ndf, opt['initial']['kernel'],\
                opt['initial']['strides'],\
                pad=opt['initial']['padding'], bn=False, act=activation[0],dpc=dpc)]
            csize,cndf = isize/2,ndf
            extra = [ConvBlock(self.ngpu,cndf, cndf, opt['extra']['kernel'], opt['extra']['strides'], bn=bn, act=activation[0],dpc=dpc) 
                     for t in range(n_extra_layers)]

            pyramid = []
            while csize > 16:
                pyramid.append(ConvBlock(self.ngpu,cndf, cndf*2, opt['pyramid']['kernel'],\
                    opt['pyramid']['strides'], pad=opt['pyramid']['padding'], bn=bn, act=activation[0],dpc=dpc))
                cndf *= 2; csize /= 2
            pyramid = pyramid+[ConvBlock(self.ngpu,cndf, cndf  , opt['extra']['kernel'],\
                opt['extra']['strides'],\
                pad=opt['extra']['padding'],  bn=bn, act=activation[0],dpc=dpc)]
            final = [Conv1d(cndf, ncl, opt['final']['kernel'],\
                opt['final']['strides'],\
                 padding=opt['final']['padding'], bias=False)]
            if bn: final = final + [BatchNorm1d(ncl)] 
            final = final + [Dpout(dpc=dpc)] + [activation[1]]
            
            self.prc = sqn(*initial)
            self.exf = extra+pyramid
            #self.exf = sqn()
            #for i,l in zip(range(extra+pyramid),extra+pyramid):
            #    self.exf.add_module('exf_{}'.format(i),Feature_extractor(l))
        if ngpu ==1:
            ann = initial+extra+pyramid+final#+[Squeeze()]+[Squeeze()]
            self.ann = sqn(*ann)
        else:
            self.ann1 = initial+extra
            self.ann2 =  pyramid+final
            self.ann1 = sqn(*self.ann1)
            self.ann2 = sqn(*self.ann2)
            self.ann1.to(self.dev0)
            self.ann2.to(self.dev1)
            
    def extraction(self,X):
        X = self.prc(X)
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            #z = pll(self.ann,X,self.gang)
            # X = self.ann1(X)
            # X = X.cuda(1)
            # X = self.ann2(X)
            import pdb
            pdb.set_trace()
            X = X.to(self.dev0)
            X = self.ann1(X)
            X = X.to(self.dev1)
            X = self.ann2(X)
            z = X
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.ann(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z
    
class DCGAN_Dz(Module):
    def __init__(self, ngpu, nz, ncl, fpd=0, n_extra_layers=1, dpc=0.0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()],bn=True,wf=False, opt=None):
        super(DCGAN_Dz,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        self.dev0 = 0
        self.dev1 = 1
        if opt is None:
            initial =[ConvBlock(self.ngpu,nz, ncl, 1, 1, bn=False, act=activation[0],dpc=dpc)]
            layers = [ConvBlock(self.ngpu,ncl,ncl, 1, 1, bn=bn, act=activation[0],dpc=dpc) 
                      for t in range(n_extra_layers)]
            
            if bn: layers = layers + [BatchNorm1d(ncl)] 
           
            #ann = initial+layers+[Dpout(dpc=dpc)]+[activation[1]]
        else:
            initial =[ConvBlock(self.ngpu,nz, ncl, 
                opt['initial']['kernel'],
                opt['initial']['strides'],
                pad=opt['initial']['padding'],
                bn=False, act=activation[0],dpc=dpc)]

            layers = [ConvBlock(self.ngpu,ncl,ncl,
                opt['extra']['kernel'],
                opt['extra']['strides'],
                pad=opt['extra']['padding'],
                bn=bn, act=activation[0],dpc=dpc) 
                      for t in range(opt['extra']['nlayers'])]
            
            if bn: layers = layers + [BatchNorm1d(ncl)] 
           
        if ngpu == 1:
            ann = initial+layers+[Dpout(dpc=dpc)]+[activation[1]]
            self.ann = sqn(*ann)
        else:
            self.ann1 = initial+layers
            self.ann2 = [Dpout(dpc=dpc)]+[activation[1]]
            self.ann1 = sqn(*self.ann1)
            self.ann2 = sqn(*self.ann2)
            self.ann1.to(self.dev0)
            self.ann2.to(self.dev1)

        self.prc = sqn(*initial)
        self.exf = layers[:-1]
    
    def extraction(self,X):
        X = self.prc(X)
        # import pdb
        # pdb.set_trace()
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            # z = pll(self.ann,X,self.gang)
            X = X.to(self.dev0)
            X = self.ann1(X)
            X = X.to(self.dev1)
            X = self.ann2(X)
            z = X
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.ann(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        
        if self.wf:
            return z,f
        else:
            return z
    
class DCGAN_G(Module):
    def __init__(self, ngpu, isize, nz, nc, ngf, n_extra_layers=0,
                 activation = [ReLU(inplace=True),Tanh()],dpc=0.0, opt=None):
        super(DCGAN_G,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        cngf, tisize = ngf//2, 4

        if opt is None:
            while tisize!=isize: cngf*=2; tisize*=2
            layers = [DeconvBlock(nz, cngf, 4, 1, 2, act=activation[0],dpc=dpc)]
            
            csize, cndf = 4, cngf
            layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 0, 0, act=activation[0],dpc=dpc))
            
            cngf //= 2; csize *= 2
            while csize < isize//2:
                layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 1,dpc=dpc))
                cngf //= 2; csize *= 2
            
            layers += [DeconvBlock(cngf, cngf, 3, 1, 1, act=activation[0],dpc=dpc) for t in range(n_extra_layers)]
            layers.append(ConvTranspose1d(cngf, nc, 4, 2, 1, bias=False))
            layers.append(activation[1])
            layers.append(Dpout(dpc=dpc))
        else:
            while tisize!=isize: cngf*=2; tisize*=2
            layers = [DeconvBlock(nz, cngf, opt['initial']['kernel'],\
                opt['initial']['strides'],\
                opt['initial']['padding'], act=activation[0],dpc=dpc)]
            
            csize, cndf = 4, cngf
            layers.append(DeconvBlock(cngf, cngf//2, opt['extra']['kernel'],\
             opt['extra']['strides'],\
              opt['extra']['padding'], 0, act=activation[0],dpc=dpc))
            
            cngf //= 2; csize *= 2
            while csize < isize//2:
                layers.append(DeconvBlock(cngf, cngf//2, opt['pyramid']['kernel'],\
                opt['pyramid']['strides'], opt['pyramid']['padding'],dpc=dpc))
                cngf //= 2; csize *= 2
            
            layers += [DeconvBlock(cngf, cngf, opt['before_end']['kernel'],\
                opt['before_end']['strides'], opt['before_end']['padding'],\
                 act=activation[0],dpc=dpc) for t in range(opt['nlayers'])]

            layers.append(ConvTranspose1d(cngf, nc, opt['final']['kernel'],\
                opt['final']['strides'], opt['final']['padding'], bias=False))
            layers.append(activation[1])
            layers.append(Dpout(dpc=dpc))

        self.ann = sqn(*layers)
            
    def forward(self,z):
        if z.is_cuda and self.ngpu > 1:
            X = pll(self.ann,z,self.gang)
        else:
            X = self.ann(z)
        if not self.training:
            X = X.detach()
        return X

class DCGAN_DXZ(Module):
    def __init__(self, ngpu, nc, n_extra_layers=0, 
                 activation=[LeakyReLU(1.0,inplace=True),LeakyReLU(1.0,inplace=True)],
                 dpc=0.0,wf=False, opt=None):
        super(DCGAN_DXZ,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        self.dev0 = 0
        self.dev1 = 1
        if opt is None:
            layers = [ConvBlock(self.ngpu,nc, nc, 1, 1, bias=True, bn=False, act=activation[0], dpc=dpc) for t in range(n_extra_layers)]
            final  = [ConvBlock(self.ngpu,nc,  1, 1, 1, bias=True, bn=False, act=activation[1], dpc=dpc)]
            ann = layers+final
            #self.exf = layers
        else:
            layers = [ConvBlock(self.ngpu,nc, nc, opt['initial']['kernel'],\
             opt['initial']['strides'],\
              pad=opt['initial']['padding'],\
               bias=True, bn=False, act=activation[0], dpc=dpc) for t in range(opt['initial']['nlayers'])]
            final  = [ConvBlock(self.ngpu,nc,  1, opt['final']['kernel'],\
             opt['final']['strides'],\
             pad=opt['final']['padding'], bias=True, bn=False, act=activation[1], dpc=dpc)]
            ann = layers+final
        if ngpu==1:
            self.exf = layers
            self.ann = sqn(*ann)
        else:
            self.ann1 = layers
            self.ann2 = final
            self.ann1 = sqn(*self.ann1)
            self.ann2 = sqn(*self.ann2) 
            self.ann1.to(self.dev0)
            self.ann2.to(self.dev1)
            
    def extraction(self,X):
        f = [self.exf[0](X)]
        for l in range(1,len(self.exf)):
            f.append(self.exf[l](f[l-1]))
        return f
    
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            #z = pll(self.ann,X,self.gang)
            X = X.to(self.dev0)
            X = self.ann1(X)
            X = X.to(self.dev1)
            X = self.ann2(X)
            z = X
            if self.wf:
                #f = pll(self.extraction,X,self.gang)
                f = self.extraction(X)
        else:
            z = self.ann(X)
            if self.wf:
                f = self.extraction(X)
        if not self.training:
            z = z.detach()
        if self.wf:
            return z,f
        else:
            return z

class DCGAN_DXX(Module):
    def __init__(self, ngpu, nc, n_extra_layers=0,
                 activation=LeakyReLU(1.0,inplace=True),dpc=0.0, opt=None):
        super(DCGAN_DXX,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        
        if opt is None:
            layers  = [ConvBlock(self.ngpu,nc, nc, 1, 1, bias=False, bn=True, act=activation, dpc=dpc) for t in range(n_extra_layers)]
            final   = [ConvBlock(self.ngpu,nc,  1, 1, 1, bias=False, bn=True, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]
        else:
            layers  = [ConvBlock(nc, nc, opt['initial']['kernel'],\
            opt['initial']['strides'],\
            pad=opt['initial']['padding'],\
            bias=False, bn=True, act=activation, dpc=dpc) for t in range(opt['nlayers'])]
            final   = [ConvBlock(nc,  1, opt['final']['kernel'],\
            opt['final']['strides'],pad=opt['final']['padding'],
            bias=False, bn=True, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]
        self.ann = sqn(*ann)
     
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
        else:
            z = self.ann(X)
        if not self.training:
            z = z.detach()
        return z

class DCGAN_DZZ(Module):
    def __init__(self, ngpu, nc, n_extra_layers=0,
                 activation=LeakyReLU(1.0,inplace=True),dpc=0.0, opt=None):
        super(DCGAN_DZZ,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        
        if opt is None:
            layers = [ConvBlock(self.ngpu,nc, nc, 1, 1, bias=False, bn=False, act=activation, dpc=dpc) for t in range(n_extra_layers)]
            final  = [ConvBlock(self.ngpu,nc,  1, 1, 1, bias=False, bn=False, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]#+[Sigmoid()]
        else:
            layers = [ConvBlock(self.ngpu,nc, nc, opt['initial']['kernel'],\
             opt['initial']['strides'],\
              pad=opt['initial']['padding'],\
               bias=False, bn=False, act=activation, dpc=dpc) for t in range(opt['layers'])]
            final  = [ConvBlock(self.ngpu,nc,  1, opt['final']['kernel'],\
             opt['final']['strides'],\
             pad=opt['final']['padding'], bias=False, bn=False, act=activation, dpc=dpc)]
            ann = layers+final+[Squeeze()]#+[Sigmoid()]
        self.ann = sqn(*ann)
     
    def forward(self,X):
        if X.is_cuda and self.ngpu > 1:
            z = pll(self.ann,X,self.gang)
        else:
            z = self.ann(X)
        if not self.training:
            z = z.detach()
        return z
