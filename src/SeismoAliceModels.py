# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''AE design'''
u'''Required modules'''
from CommonNN import *
import torch
from CommonTorch import tdev
from math import ceil
# class DenseEncoder(Module):
#     def __init__(self,ngpu,dev,ninp,nout,szs,nly,
#                  act=[LeakyReLU(1.0,True),Tanh()],dpc=0.10):
#         super(DenseEncoder,self).__init__()
#         self.ngpu = ngpu
#         self.gang = range(self.ngpu)
#         self.dev = dev
#         self.cnn = []
#         for l in range(nly):
#             self.cnn.append(ConvBlock(ni=ninp[l],no=nout[l],ks=szs-1,
#                                       stride=1,bias=True,act=act[l],
#                                       bn=False))
#         self.cnn = sqn(*self.cnn)

#     def forward(self,Xn):
#         if Xn.is_cuda and self.ngpu > 1:
#             zlf   = pll(self.cnn,Xn,self.gang)
#         else:
#             zlf   = self.cnn(Xn)
#         if not self.training:
#             zlf=zlf.detach()
#         return zlf
    
# class ResEncoder(Module):
#     def __init__(self,ngpu,dev,nz,nzcl,nch,ndf,\
#                  szs,nly,ker=7,std=4,pad=0,dil=1,grp=1,\
#                  dpc=0.10,act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),LeakyReLU(1.0,True),Tanh()],\
#                  with_noise=False,dtm=0.01,ffr=0.16,wpc=2.e-2):
#         super(ResEncoder,self).__init__()
#         self.ngpu = ngpu
#         self.gang = range(self.ngpu)
#         self.dev = dev
        
#         self.cnn = [ConvBlock(ni=nch*1,no=ndf*2,ks=ker,stride=std,act=act[0],bn=False),
#                     ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[1],bn=True ,dpc=dpc),
#                     ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[2],bn=True ,dpc=dpc),
#                     ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[3],bn=True ,dpc=dpc),
#                     ResConvBlock(ni=ndf*2,no=ndf*2,ks=3,stride=1,act=act[4],bn=True,dpc=dpc)]
        
#         self.cnn = [ResNet(ann=self.cnn),ConvBlock(ni=ndf*2,no=nz,ks=33,pad=16,stride=16,act=act[-1],bn=False)]
        
#         self.cnn = sqn(*self.cnn)
        
#     def forward(self,Xn):
#         if Xn.is_cuda and self.ngpu > 1:
#             zlf   = pll(self.cnn,Xn,self.gang)
#         else:
#             zlf   = self.cnn(Xn)
#         if not self.training:
#             zlf=zlf.detach()
#         return zlf

# class EncoderDecoder(Module):
#     def __init__(self,ngpu,dev,nz,nch,ndf,nly,ker=7,std=4,opd=0,pad=0,dil=1,grp=1,bn=True,
#                  dpc=0.10,act=[LeakyReLU(1.0,True),LeakyReLU(1.0,True),
#                                LeakyReLU(1.0,True),LeakyReLU(1.0,True),
#                                LeakyReLU(1.0,True),Tanh()]):
#         super(EncoderDecoder,self).__init__()
#         self.ngpu = ngpu
#         self.gang = range(self.ngpu)
#         self.dev = dev
#         if nly==3:
#             self.cnn = \
#                 cnn1d(nch*2,ndf*1,act[0],ker=ker,std=std,pad=pad+1,bn=bn,dpc=dpc  ,wn=False)+\
#                 cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad+1,bn=bn,dpc=dpc  ,wn=False)+\
#                 cnn1d(ndf*2,nz ,act[2],ker=ker,std=std,pad=pad+1,bn=False,dpc=0.0 ,wn=True,dev=self.dev)+\
#                 cnn1dt(2*nz ,ndf*2,act[3],ker=ker,std=std,pad=1,opd=opd,bn=True, dpc=dpc)+\
#                 cnn1dt(ndf*2,ndf*1,act[4],ker=ker,std=std,pad=1,opd=opd,bn=True, dpc=dpc)+\
#                 cnn1dt(ndf*1,nch*1,act[5],ker=ker,std=std,pad=1,opd=opd,bn=False, dpc=0.0)
#         elif nly==5:
#             self.cnn = \
#                 cnn1d(nch*2,ndf*1,act[0],ker=ker,std=std,pad=pad,bn=bn,dpc=0.0   ,wn=False)+\
#                 cnn1d(ndf*1,ndf*2,act[1],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc   ,wn=False)+\
#                 cnn1d(ndf*2,ndf*4,act[2],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc   ,wn=False)+\
#                 cnn1d(ndf*4,ndf*8,act[3],ker=ker,std=std,pad=pad,bn=bn,dpc=dpc   ,wn=False)+\
#                 cnn1d(ndf*8,nz   ,act[4],ker=ker,std=std,pad=2  ,bn=False,dpc=0.0,wn=True)+\
#                 cnn1dt(2*nz,ndf*8,act[5],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
#                 cnn1dt(ndf*8,ndf*4,act[6],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
#                 cnn1dt(ndf*4,ndf*2,act[7],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
#                 cnn1dt(ndf*2,ndf*1,act[8],ker=ker,std=std,pad=1,opd=opd,bn=True , dpc=dpc)+\
#                 cnn1dt(ndf*1,nch*1,act[9],ker=ker,std=std,pad=1,opd=opd,bn=False, dpc=0.0)
#         self.cnn = sqn(*self.cnn)

#     def forward(self,Xn):
#         if Xn.is_cuda and self.ngpu > 1:
#             zlf   = pll(self.cnn,Xn,self.gang)
#         else:
#             zlf   = self.cnn(Xn)
#         if not self.training:
#             zlf=zlf.detach()
#         return zlf

# sum(p.numel() for p in self.parameters())

class Encoder(Module):
    def __init__(self,d_x,d_z,**config):
        super(Encoder,self).__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.h = []

    def compile(self,config):
        self.h = Sequential(self.h)
        pass

    def get(self,net_type,**config):
        if net_type=="branched":
            net = BranchedEncoder(self.d_x,self.d_z,**config)
        net.compile()
        return net

    def forward(self,x):
        z = self.h(x)
        return z if self.training else z.detach()

class BranchedEncoder(Encoder):
    def __init__(self,d_x,d_z,config):
        super(BranchedEncoder,self).__init__(d_x,d_z)
        self.config = config
        self.hxy = []
        self.hyy = []

    def check_dimensions(self):
        self.Lout = {}
        self.Lout.fromkeys(self.config.keys(),[])
        self.Lout["Fin"] = [self.d_x]

        k,v=("Fin",self.config["Fin"])

        for l in range(v["nlayers"]):
            if "Conv1d" in v["layers"][l]:
                if "padding" not in self.config[k].keys():
                    self.config[k]["padding"]=[]

                if len(self.config[k]["padding"])<self.config[k]["nlayers"]:
                    self.config[k]["padding"].append(ceil((
                        self.config[k]["dilation"][l]*
                        (self.config[k]["kernel_size"][l]-1)+1-
                        self.config[k]["stride"][l])/2))

                self.Lout[k].append(int(1+(self.Lout[k][l]+
                    2*self.config[k]["padding"][l]-
                        self.config[k]["dilation"][l]*(
                            self.config[k]["kernel_size"][l]-1
                        )-1)/self.config[k]["stride"][l])
                    )
            elif "Flat" in v["layers"][l]:
                self.config[k]["padding"].append(0)
                self.Lout[k].append(self.Lout[k][-1]*self.config[k]["channels"][l])
                self.config[k]["channels"][l+1] = self.Lout[k][-1]

            elif "Linear" in v["layers"][l]:
                self.config[k]["padding"].append(0)
                self.Lout[k].append(-1)

        for k,v in self.config.items():
            if k!="Fin":
                self.Lout[k] = [self.Lout["Fin"][-1]]
                for l in range(v["nlayers"]):
                    if "Conv1d" in v["layers"][l]:
                        if "padding" not in self.config[k].keys():
                            self.config[k]["padding"]=[]
                        if len(self.config[k]["padding"])<self.config[k]["nlayers"]:
                            self.config[k]["padding"].append(ceil((
                                self.config[k]["dilation"][l]*
                                (self.config[k]["kernel_size"][l]-1)+1-
                                self.config[k]["stride"][l])/2))

                        self.Lout[k].append(int(1+(self.Lout[k][l]+
                                2*self.config[k]["padding"][l]-
                                self.config[k]["dilation"][l]*(
                                    self.config[k]["kernel_size"][l]-1
                                )-1)/self.config[k]["stride"][l])
                            )
                    elif "Flat" in v["layers"][l]:
                        self.config[k]["padding"].append(0)
                        self.Lout[k].append(self.Lout[k][-1]*self.config[k]["channels"][l])
                        self.config[k]["channels"][l+1] = self.Lout[k][-1]

                    elif "Linear" in v["layers"][l]:
                        self.config[k]["padding"].append(0)
                        self.Lout[k].append(-1)
        return

    def compile(self):

        self.check_dimensions()

        self.h = [eval(self.config["Fin"]["layers"][l]+
            self.config["Fin"]["postconv"][l])(
            in_channels=self.config["Fin"]["channels"][l],
            out_channels=self.config["Fin"]["channels"][l+1],
            batchnorm=eval(self.config["Fin"]["batchnorm"])[l],
            activation=eval(self.config["Fin"]["activation"])[l],
            dropout=self.config["Fin"]["dropout"][l],
            bias=eval(self.config["Fin"]["bias"])[l],
            kernel_size=self.config["Fin"]["kernel_size"][l],
            stride=self.config["Fin"]["stride"][l],
            padding=self.config["Fin"]["padding"][l],
            dilation=self.config["Fin"]["dilation"][l]
            )
            for l in range(self.config["Fin"]["nlayers"])]

        self.hxy = [eval(self.config["Fxy"]["layers"][l]+
            self.config["Fxy"]["postconv"][l])(
            in_channels=self.config["Fxy"]["channels"][l],
            out_channels=self.config["Fxy"]["channels"][l+1],
            batchnorm=eval(self.config["Fxy"]["batchnorm"])[l],
            activation=eval(self.config["Fxy"]["activation"])[l],
            dropout=self.config["Fxy"]["dropout"][l],
            bias=eval(self.config["Fxy"]["bias"])[l],
            kernel_size=self.config["Fxy"]["kernel_size"][l],
            stride=self.config["Fxy"]["stride"][l],
            padding=self.config["Fxy"]["padding"][l],
            dilation=self.config["Fxy"]["dilation"][l]
            )
            for l in range(self.config["Fxy"]["nlayers"])]

        self.hyy = [eval(self.config["Fyy"]["layers"][l]+
            self.config["Fyy"]["postconv"][l])(
            in_channels=self.config["Fyy"]["channels"][l],
            out_channels=self.config["Fyy"]["channels"][l+1],
            batchnorm=eval(self.config["Fyy"]["batchnorm"])[l],
            activation=eval(self.config["Fyy"]["activation"])[l],
            dropout=self.config["Fyy"]["dropout"][l],
            bias=eval(self.config["Fyy"]["bias"])[l],
            kernel_size=self.config["Fyy"]["kernel_size"][l],
            stride=self.config["Fyy"]["stride"][l],
            padding=self.config["Fyy"]["padding"][l],
            dilation=self.config["Fyy"]["dilation"][l]
            )
            for l in range(self.config["Fyy"]["nlayers"])]

        self.h = Sequential(*self.h)
        self.hxy = Sequential(*self.hxy)
        self.hyy = Sequential(*self.hyy)

    def forward(self,x):
        z = self.h(x)
        zxy = self.hxy(z)
        zyy = self.hyy(z)
        return (zxy,zyy) if self.training else (zxy.detach(),zyy.detach())

    def get(self,*args,**kwargs):
        return self




    def forward(self,x):
        z = self.h(x)
        return z if self.training else z.detach()

class DCGAN_D(Module):
    def __init__(self, ngpu, isize, nc, ncl, ndf, n_extra_layers=0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()]):
        super(DCGAN_D,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        initial = [ConvBlock(nc, ndf, 4, 2, bn=False, act=activation[0])]
        
        csize,cndf = isize/2,ndf
        
        extra = [ConvBlock(cndf, cndf, 3, 1, act=activation[0]) 
                 for t in range(n_extra_layers)]

        pyramid = []
        while csize > 4:
            pyramid.append(ConvBlock(cndf, cndf*2, 4, 2,act=activation[0]))
            cndf *= 2; csize /= 2
        
        final = [Conv1d(cndf, ncl, 4, padding=0, bias=False)]
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
                 activation=[LeakyReLU(1.0,True),Sigmoid()],bn=True,wf=False):
        super(DCGAN_Dx,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
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
    def __init__(self, ngpu, nz, ncl, fpd=0, n_extra_layers=1, dpc=0.0,
                 activation=[LeakyReLU(1.0,True),Sigmoid()],bn=True,wf=False):
        super(DCGAN_Dz,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        initial =[ConvBlock(nz, ncl, 1, 1, bn=False, act=activation[0],dpc=dpc)]
        layers = [ConvBlock(ncl,ncl, 1, 1, bn=bn, act=activation[0],dpc=dpc) 
                  for t in range(n_extra_layers)]
        
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
                 activation = [ReLU(inplace=True),Tanh()],dpc=0.0):
        super(DCGAN_G,self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        cngf, tisize = ngf//2, 4
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
                 dpc=0.0,wf=False):
        super(DCGAN_DXZ,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        self.wf = wf
        layers = [ConvBlock(nc, nc, 1, 1, bias=True, bn=False, act=activation[0], dpc=dpc) for t in range(n_extra_layers)]
        final  = [ConvBlock(nc,  1, 1, 1, bias=True, bn=False, act=activation[1], dpc=dpc)]
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
                 activation=LeakyReLU(1.0,inplace=True),dpc=0.0):
        super(DCGAN_DXX,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        
        layers  = [ConvBlock(nc, nc, 1, 1, bias=False, bn=True, act=activation, dpc=dpc) for t in range(n_extra_layers)]
        final   = [ConvBlock(nc,  1, 1, 1, bias=False, bn=True, act=activation, dpc=dpc)]
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
                 activation=LeakyReLU(1.0,inplace=True),dpc=0.0):
        super(DCGAN_DZZ,self).__init__()
        self.ngpu = ngpu
        self.gang = range(self.ngpu)
        
        layers = [ConvBlock(nc, nc, 1, 1, bias=False, bn=False, act=activation, dpc=dpc) for t in range(n_extra_layers)]
        final  = [ConvBlock(nc,  1, 1, 1, bias=False, bn=False, act=activation, dpc=dpc)]
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
