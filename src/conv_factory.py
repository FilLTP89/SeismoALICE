from encoder_modelparallele import *
from encoder_dataparallele import *
from decoder_modelparallele import *
from decoder_dataparallele import *
from dcgan_dzmodelparallele import *
from dcgan_dxzmodelparallele import *
from dcgan_dxmodelparallele import *
from dcgan_dxdataparallele import *
from dcgan_dzdataparallele import *
from dcgan_dxzdataparallele import *

from profile_support import profile

class ConvNetFactory(object):
    """
    ConvNetFactory is to generate Encoder, Decoder and Discriminator method
    The following code is a Abstract Factory Pattern. We generate 2 strategy:
    1. Model Parallele
    2. Data Parallele
    """
    def __init__(self):
        super(ConvNetFactory, self).__init__()
        pass

    def createEncoder():
        pass

    def createDecoder():
        pass

    def createDCGAN_Dx():
        pass

    def createDCGAN_Dz():
        pass

    def createDCGAN_DXZ():
        pass


class ModelParalleleFactory(ConvNetFactory):
    """act
    ModelParalleleFactory is implement the ConVNetFactory methodes
    This class is responsible to generate the Model Parallel strategy
    of Pytorch. Encoder, Decoder and Discriminators will be generated based on this approach
    """
    def __init__(self):
        super(ModelParalleleFactory, self).__init__()
        pass
    
    def createEncoder(self, config, opt, *args, **kwargs):

        return EncoderModelParallele.getEncoderByGPU(ngpu=opt.ngpu, dev = opt.dev,\
                nz  = opt.nzd, nch=opt.nch, ndf = opt.ndf,\
                nly = config['nlayers'],\
                ker = config['kernel'],\
                dil = config['dilation'],\
                std = config['strides'],\
                pad = config['padding'],
                act = config['act'],\
                limit = config['limit'],\
                channel = config['channel'],\
                dpc = 0.0,\
                *args, **kwargs)
    @profile
    def createDecoder(self, config, opt, *args, **kwargs):
        return DecoderModelParallele.getDecoderByGPU(ngpu=opt.ngpu,nz=opt.nzd,\
                nch=opt.nch, ndf=opt.ndf,\
                nly=config['nlayers'],\
                ker = config['kernel'],\
                std = config['strides'],\
                dil = config['dilation'],\
                pad = config['padding'],\
                opd = config['outpads'],\
                act = config['act'],\
                limit = config['limit'],\
                channel = config['channel'],\
                bn = True,\
                dpc = 0.0,\
                n_extra_layers=0,\
                *args, **kwargs)

    def createDCGAN_Dx(self, config_dcgan_dx, opt, *args, **kwargs):
        nc  = config_dcgan_dx["nc"] if "nc" in config_dcgan_dx else opt.nch
        dcgan_dx  = DCGAN_DxModelParallele.getDCGAN_DxByGPU(ngpu=opt.ngpu, isize=256,\
                                     nc = nc, ncl=512, ndf=opt.ndf, fpd=1,\
                                     nly = config_dcgan_dx['nlayers'],\
                                     ker = config_dcgan_dx['kernel'],\
                                     std = config_dcgan_dx['strides'],\
                                     pad = config_dcgan_dx['padding'],\
                                     dil = config_dcgan_dx['dilation'],\
                                     act = config_dcgan_dx['act'],\
                                     limit = config_dcgan_dx['limit'],\
                                     channel = config_dcgan_dx['channel'],\
                                     grp=1,bn=False,wf=False, dpc=0.40,
                                     n_extra_layers=1)
        return dcgan_dx

    def createDCGAN_Dz(self, config_dcgan_dz, opt, *args, **kwargs):
        nz    = config_dcgan_dz["nz"]   if "nz" in config_dcgan_dz else opt.nzd
        dcgan_dz  = DCGAN_DzModelParallele.getDCGAN_DzByGPU(ngpu=opt.ngpu, nz=nz,\
                                     ncl=1024, fpd=0, n_extra_layers=0, dpc=0.40,
                                     nly = config_dcgan_dz['nlayers'],\
                                     ker = config_dcgan_dz['kernel'],\
                                     std = config_dcgan_dz['strides'],\
                                     pad = config_dcgan_dz['padding'],\
                                     act = config_dcgan_dz['act'],\
                                     limit = config_dcgan_dz['limit'],\
                                     channel = config_dcgan_dz['channel'],\
                                     dil= config_dcgan_dz["dilation"],\
                                     grp=1,bn=False,wf=False, bias = False)
        return dcgan_dz

    def createDCGAN_DXZ(self,config_dcgan_dxz, opt, *args, **kwargs):
        nc    = config_dcgan_dxz["nc"] if "nc" in config_dcgan_dxz else 1024
        dcgan_dxz = DCGAN_DXZModelParallele.getDCGAN_DXZByGPU(ngpu=opt.ngpu, nc = nc,\
                                     nly = config_dcgan_dxz['nlayers'],\
                                     ker = config_dcgan_dxz['kernel'],\
                                     std = config_dcgan_dxz['strides'],\
                                     pad = config_dcgan_dxz['padding'],\
                                     act = config_dcgan_dxz['act'],\
                                     limit = config_dcgan_dxz['limit'],\
                                     dil = config_dcgan_dxz['dilation'],\
                                     channel = config_dcgan_dxz['channel'],\
                                     n_extra_layers=0,
                                     dpc=0.40,wf=False,bn = False, bias = True, opt=None)
        return  dcgan_dxz

class DataParalleleFactory(ConvNetFactory):
    """
    DataParalleleFactory implement the ConvNetFactory methods
    This class is reponsible to generate the DataParallele strategy for 
    Encoder, Decoder and Discriminators classes. 
    """
    def __init__(self):
        super(DataParalleleFactory, self).__init__()
        pass

    def createEncoder(self, config, opt, *args, **kwargs):
        
        name  = config["name"] if hasattr(config, "name") else None
        return EncoderDataParallele.getEncoder(name = name, ngpu = opt.ngpu,\
                dev = opt.dev,\
                nz = opt.nzd, nch = opt.nch,\
                ndf = opt.ndf,\
                nly = config['nlayers'],\
                ker = config['kernel'],\
                dil = config['dilation'],\
                std = config['strides'],\
                pad = config['padding'],\
                act = config['act'],\
                limit = config['limit'],\
                channel = config['channel'],\
                dpc = 0.0,\
                *args, **kwargs)

    def createDecoder(self, config, opt, *args, **kwargs):
        name  = config["name"] if hasattr(config, "name") else None
        return DecoderDataParallele.getDecoder(name = name, ngpu = opt.ngpu,\
                nz = opt.nzd, nch = opt.nch,\
                ndf = opt.ndf,\
                nly = config['nlayers'],\
                ker = config['kernel'],\
                std = config['strides'],\
                dil = config['dilation'],\
                pad = config['padding'],\
                opd = config['outpads'],\
                act = config['act'],\
                limit = config['limit'],\
                channel = config['channel'],\
                dpc = 0.0\
                *args, **kwargs)

    def createDCGAN_Dx(self, config_dcgan_dx, opt, *args, **kwargs):

        #another class could be called here if is not the generic class
        name  = config_dcgan_dx["name"] if "name" in config_dcgan_dx else None
        nc  = config_dcgan_dx["nc"] if "nc" in config_dcgan_dx else opt.nch
        #DCGAN_Dx class is called here
        dcgan_dx    = DCGAN_DxDataParallele.getDCGAN_DxDataParallele(name=name, ngpu = opt.ngpu,\
                     isize= 256, nc = nc,\
                     ncl = 512, ndf = opt.ndf, fpd=1,\
                     nly = config_dcgan_dx['nlayers'],\
                     ker = config_dcgan_dx['kernel'],\
                     std = config_dcgan_dx['strides'],\
                     pad = config_dcgan_dx['padding'],\
                     act = config_dcgan_dx['act'],\
                     limit = config_dcgan_dx['limit'],\
                     dil = config_dcgan_dx['dilation'],\
                     channel = config_dcgan_dx['channel'],\
                     grp=0,bn=False,wf=False, dpc=0.40,\
                     n_extra_layers=0)
        return dcgan_dx

    def createDCGAN_Dz(self, config_dcgan_dz, opt, *args, **kwargs):
        #DCGAN_DZ class is called here 
        name  = config_dcgan_dz["name"] if "name" in config_dcgan_dz else None
        nz    = config_dcgan_dz["nz"]   if "nz" in config_dcgan_dz else opt.nzd
        dcgan_dz    = DCGAN_DzDataParallele.getDCGAN_DzDataParallele( name =  name, ngpu = opt.ngpu,\
                     nc=opt.nch, nz = nz,\
                     ncl=512, ndf=opt.ndf, fpd=0,\
                     nly = config_dcgan_dz['nlayers'],\
                     ker = config_dcgan_dz['kernel'],\
                     std = config_dcgan_dz['strides'],\
                     pad = config_dcgan_dz['padding'],\
                     act = config_dcgan_dz['act'],\
                     limit = config_dcgan_dz['limit'],\
                     dil = config_dcgan_dz['dilation'],\
                     channel = config_dcgan_dz['channel'],\
                     grp =0, bn=False,wf=False, dpc=0.40,
                     n_extra_layers=0)
        return dcgan_dz

    def createDCGAN_DXZ(self, config_dcgan_dxz,opt, *args, **kwargs):
        #DCGAN_DXZ class is called here
        name  = config_dcgan_dxz["name"] if "name" in config_dcgan_dxz else None
        nc    = config_dcgan_dxz["nc"] if "nc" in config_dcgan_dxz else 1024
        dcgan_dxz   = DCGAN_DXZDataParallele.getDCGAN_DXZDataParallele(name = name, ngpu=opt.ngpu, nc=nc, 
                     nly=config_dcgan_dxz['nlayers'],\
                     ker=config_dcgan_dxz['kernel'],\
                     std=config_dcgan_dxz['strides'],\
                     pad=config_dcgan_dxz['padding'],\
                     act=config_dcgan_dxz['act'],\
                     limit = config_dcgan_dxz['limit'],\
                     dil=config_dcgan_dxz['dilation'],\
                     channel = config_dcgan_dxz['channel'],\
                     grp=0, bn=False,wf=False, dpc=0.40,\
                     n_extra_layers=0)

        return dcgan_dxz


    
class Network(object):
    """
    Network take in charge to return the right encoder, decoder and discriminators
    for the environnement
    """
    def __init__(self, ConvNetFactory):
        super(Network, self).__init__()
        self.ConvNetFactory = ConvNetFactory

    def Encoder(self, config, opt, *args, **kwargs):
        return self.ConvNetFactory.createEncoder(config, opt, *args, **kwargs)

    def Decoder(self, config, opt, *args, **kwargs):
        return self.ConvNetFactory.createDecoder(config, opt, *args, **kwargs)

    def DCGAN_Dx(self, config, opt, *args, **kwargs):
        return self.ConvNetFactory.createDCGAN_Dx(config, opt, *args, **kwargs)

    def DCGAN_Dz(self, config, opt, *args, **kwargs):
        return self.ConvNetFactory.createDCGAN_Dz(config, opt, *args, **kwargs)

    def DCGAN_DXZ(self, config, opt, *args, **kwargs):
        return self.ConvNetFactory.createDCGAN_DXZ(config, opt, *args, **kwargs)


