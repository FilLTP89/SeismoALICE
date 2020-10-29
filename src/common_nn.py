# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
# COMMON
from common_model import AKA
# NUMPY
import numpy as np
# TORCH GENERIC
#from torch._jit_internal import weak_module, weak_script_method
# NN GENERIC
import torch
from torch import FloatTensor as tFT
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torch.nn import Sequential as sqn
from torch.nn import Linear as lin
from torch.nn.parallel import data_parallel as pll
from torch.nn import DataParallel
from torch import device as tdev
# LOSSES
from torch.nn.modules.loss import _Loss
from torch.nn import MSELoss as MSE
from torch.nn import NLLLoss as NLL
from torch.nn import BCELoss as BCE
from torch.nn import CrossEntropyLoss as CEL
# OPTIMIZER
from torch.optim import Adam, RMSprop, SGD
from itertools import repeat as ittr
from itertools import chain as ittc
# AUTOGRAD
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# COMMON TORCH
from common_torch import tfnp,tcat,tavg
from common_torch import trnd,ln0c,tnrm,ttns
# NOISE 
from generate_noise import noise_generator
rndm_args = {'mean': 0, 'std': 1}
eps = 1e-15  # to avoid possible numerical instabilities during backward
b1 = 0.5
b2 = 0.9999
  
# def get_features(t_img,model,layer,szs):
#     
#      Create a vector of zeros that will hold our feature vector
#     my_embedding = torch.zeros(szs)
#      Define a function that will copy the output of a layer
#     def copy_data(m,i,o):
#         my_embedding.copy_(m.bckward_pass(o).data)
#      Attach that function to our selected layer
#     h = layer.register_forward_hook(copy_data)
#      Run the model on our transformed image
#     model(t_img)
#      Detach our copy function from the layer
#     h.remove()
#     return my_embedding
# 
# class FeatureExtractor(Module):
#     def __init__(self, cnn, feature_layer=11):
#         super(FeatureExtractor, self).__init__()
#         self.features = sqn(*list(cnn.features.children())[:(feature_layer+1)])
#     def forward(self, x):
#         return self.features(x)
# class LayerActivations():    
#     features=[]        
#     def __init__(self,model):        
#         self.features = []        
#         self.hook = model.register_forward_hook(self.hook_fn)        
#         def hook_fn(self,module,input,output):                
#             self.features.extend(output.view(output.size(0),-1).cpu().data)        
#         def remove(self):                
#             self.hook.remove()
        
class Feature_extractor(Module):
    def __init__(self,lay):
        super(Feature_extractor,self).__init__()
        self.lay=lay
    def forward(self,X):
        self.feature = self.lay(X)
        return X
  
# Add noise module
class AddNoise(Module):
    def __init__(self,dev=tdev("cpu")):
        super(AddNoise,self).__init__()
        self.dev = dev
    def forward(self,X):
        W,_,_ = noise_generator(X.shape,X.shape,self.dev,rndm_args)
        #if X.is_cuda:
        #    .cuda()
        #else:
        #    return zcat(X,W)
        return zcat(X,W) 
class Swish(Module):
    def __init__(self, train_beta=False):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(torch.Tensor([1.]))
        else:
            self.weight = 1.0

    def forward(self, x):
        return x * torch.sigmoid(self.weight * x)
    
# Dropout module
class Dpout(Module):
    def __init__(self,dpc=0.10):
        super(Dpout,self).__init__()
        self.dpc = dpc
    def forward(self,x):
        return F.dropout(x,p=self.dpc,\
                         training=self.training)

class Flatten(Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        return x.view(-1,1).squeeze(1)

class Squeeze(Module):
    def __init__(self):
        super(Squeeze,self).__init__()
    def forward(self,x):
        return x.squeeze(1)

def cnn1d(in_channels,out_channels,\
          act=LeakyReLU(1.0,inplace=True),\
          bn=True,ker=7,std=4,pad=1,\
          dil=1,grp=1,dpc=0.1,wn=False,dev=tdev("cpu")):

    block = [Conv1d(in_channels=in_channels,\
                    out_channels=out_channels,\
                    kernel_size=ker,stride=std,\
                    padding=pad,dilation=dil,groups=grp,\
                    bias=False)]
    #if wn:
    #    block.insert(0,AddNoise(dev=dev))
    if bn:
        block.append(BatchNorm1d(out_channels))
    block.append(act)
    block.append(Dpout(dpc=dpc))
    if wn:
        block.append(AddNoise(dev=dev))
    return block

def cnn1dt(in_channels,out_channels,\
           act=LeakyReLU(1.0,inplace=True),\
           bn=True,ker=7,std=4,pad=0,opd=0,\
           dil=1,grp=1,dpc=0.1):

    block = [ConvTranspose1d(in_channels=in_channels,\
                             out_channels=out_channels,\
                             kernel_size=ker,stride=std,\
                             output_padding=opd,padding=pad,\
                             dilation=dil,groups=grp,\
                             bias=False)]
    if bn:
        block.append(BatchNorm1d(out_channels))
    block.append(act)
    block.append(Dpout(dpc=dpc))
    return block

def DenseBlock(in_channels,out_channels,\
               act=[Sigmoid()],dpc=0.1):

    block = [Linear(in_channels,\
                       out_channels,\
                       bias=False),
            act,Dpout(dpc=dpc)]
    
    return block

class ConvBlock(Module):
    def __init__(self,ngpu, ni, no, ks, stride, bias=False,
                 act = None, bn=True, pad=None, dpc=None):
        super(ConvBlock,self).__init__()
        self.ngpu = ngpu
        if pad is None: pad = ks//2//stride
        self.ann = [Conv1d(ni, no, ks, stride, padding=pad, bias=bias)]
        if bn: self.ann+= [BatchNorm1d(no)]
        if dpc is not None: self.ann += [Dpout(dpc=dpc)]
        if act is not None: self.ann += [act] 
         
        self.ann = sqn(*self.ann)
    
    def forward(self, X):
        if X.is_cuda and self.ngpu > 1:
            #z = pll(self.ann,X,self.gang)
            # X = X.to(self.dev0)
            # X = self.ann1(X)
            # X = X.to(self.dev1)
            X = self.ann(X)
            z = X
            return z
        else:
            return self.ann(x)

class DeconvBlock(Module):
    def __init__(self,ni,no,ks,stride,pad,opd=0,bn=True,act=ReLU(inplace=True),
                 dpc=None):
        super(DeconvBlock,self).__init__()
        self.conv = ConvTranspose1d(ni, no, ks, stride, 
                                    padding=pad, output_padding=opd,
                                    bias=False)
        self.bn = BatchNorm1d(no)
        self.relu = act
        self.dpout = Dpout(dpc=dpc)
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.dpout(self.bn(x)) if self.bn else self.dpout(x)
    
class ResConvBlock(Module):
    def __init__(self, ni, no, ks, stride, bias=False,
                 act = None, bn=True, pad=None, dpc=None):
        super(ResConvBlock,self).__init__()
        
        # 1st block
        if pad is None: pad = ks//2//stride
        self.ann = [Conv1d(ni, no, ks, stride, padding=pad, bias=bias)]
        if bn: self.ann += [BatchNorm1d(no)]
        if dpc is not None: self.ann += [Dpout(dpc=dpc)]
        
        # 2nd block
#         stride = 1
#         ks = ks-1
#         if pad is None: pad = ks//2//stride
        self.ann += [Conv1d(no, no, ks, stride, padding=pad, bias=bias)]
        if bn: self.ann += [BatchNorm1d(no)]
        if dpc is not None: self.ann += [Dpout(dpc=dpc)]
        # activation
        if act is not None: self.ann += [act]

        self.ann = sqn(*self.ann)
        
    def forward(self, x):
        return self.ann(x) + x

class ResNet(Module):
    def __init__(self, ann):
        super(ResNet,self).__init__()
        
        # 1st block
        self.ann = sqn(*ann)
        
    def forward(self, x):
        return self.ann(x) + self.ann._modules['0'](x)
    
u'''[Zeroed gradient for selected optimizers]'''
def zerograd(optz):
    for o in optz: 
        if o is not None:
            o.zero_grad()
        
def penalty(loss,params,typ,lam=1.e-5):
    pen = {'L1':1,'L2':2}
    reg = ttns(0.)
    for p in params:
        reg += tnrm(p,pen[typ])
    return loss + lam*reg

def gan_loss(yp,yt,reduction=True):
    if reduction:
        return -tavg(ln0c(yt.cpu()) + ln0c(1.0-yp.cpu()))
    else:
        return -ln0c(yt.cpu()) - ln0c(1.0-yp.cpu())
    
def softplus(_x):
    return torch.log(1.0 + torch.exp(_x))
    
#@weak_module
class GANLoss(_Loss):
    __constants__ = ['reduction']
    __metaclass__ = AKA
    aliases = 'GAN'
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(GANLoss, self).__init__(size_average, reduce, reduction)

    #@weak_script_method
    def forward(self, inp,tar):
        return gan_loss(inp,tar,reduction=self.reduction)
GAN = GANLoss

#@weak_module
class ALILoss(_Loss):
    __metaclass__ = AKA
    aliases = 'ALI'
    def __init__(self, dloss = True):
        super(ALILoss, self).__init__(dloss)
        self.dloss=dloss
    #@weak_script_method
    def forward(self,sample_preds,data_preds):
        if self.dloss:
            # discriminator loss
            return torch.mean(softplus(-data_preds-eps) + softplus(sample_preds+eps))
        else:
            # generator loss
            return torch.mean(softplus(data_preds+eps) + softplus(-sample_preds-eps))

ALI = ALILoss

def zcat(*args):
    return tcat(args,1)

# custom weights initialization called on netG and netD¬                 
def set_weights(m):                                 
    classname = m.__class__.__name__               
    if classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:                   
        try:
            init.xavier_uniform(m.weight)
        except:
            print("OK")
        #m.weight.data.normal_(0.0, 0.02)                                
    elif classname.find('BatchNorm') != -1:                             
        m.weight.data.normal_(1.0, 0.02)                              
        m.bias.data.fill_(0)

def tie_weights(m):
    for _,n in m.__dict__['_modules'].items():
        try:
            n.bckward_pass[0].weight = n.forward_pass[0].weight
        except: 
            pass

def reset_net(nets,func=set_weights,lr=0.0002,b1=b1,b2=b2,weight_decay=None,
              optim='Adam'):
    p = []
    for n in nets:
        n.apply(func)
        p.append(n.parameters())
    if 'adam' in optim.lower():
        if  weight_decay is None:
            return Adam(ittc(*p),lr=lr,betas=(b1,b2))
        else:
            return Adam(ittc(*p),lr=lr,betas=(b1,b2),weight_decay=weight_decay) 
    elif 'rmsprop' in optim.lower():
        return RMSprop(ittc(*p),lr=lr)
    elif 'sgd' in optim.lower():
        return SGD(ittc(*p),lr=lr)


def clipweights(netlist,lb=-0.01,ub=0.01):
    for D in netlist:
        for p in D.parameters():
            p.data.clamp_(lb,ub)
    
def dump_conf(m):                                 
    classname = m.__class__.__name__               
    if classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:
        print('cuda-check')                       
        print(classname)
        print(m.weight.is_cuda)

#def zero_dpout(m):
#    for name, param in m.named_children():
#        if isinstance(param,Sequential):
#            import pdb
#            pdb.set_trace()

def to_categorical(y,c2type,num_columns=1):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.
    return Variable(c2type(y_cat))

def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = tfnp(cat)
    return Variable(cat)

def runout(funct, world_size):
    mp.spawn(funct,
             args=(world_size,),
             nprocs=world_size,
             join=True)

