import json
from net.encoder.encoder_modelparallele import EncoderModelParallele
from net.encoder.encoder_dataparallele  import EncoderDataParallele
from net.decoder.decoder_modelparallele import DecoderModelParallele
from net.decoder.decoder_dataparallele  import DecoderDataParallele
from net.dcgan.dcgan_dzmodelparallele   import DCGAN_DzModelParallele
from net.dcgan.dcgan_dxzmodelparallele  import DCGAN_DXZModelParallele
from net.dcgan.dcgan_dxmodelparallele   import DCGAN_DxModelParallele
from net.dcgan.dcgan_dxdataparallele    import DCGAN_DxDataParallele
from net.dcgan.dcgan_dzdataparallele    import DCGAN_DzDataParallele
from net.dcgan.dcgan_dxzdataparallele   import DCGAN_DXZDataParallele
# from profiling.profile_support import profile




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
        raise NotImplementedError

    def createDecoder():
        raise NotImplementedError

    def createDCGAN_Dx():
        raise NotImplementedError

    def createDCGAN_Dz():
        raise NotImplementedError

    def createDCGAN_DXZ():
        raise NotImplementedError


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
        path  = config["path"] if "path" in config else ""
        dconv = config["dconv"] if "dconv" in config else ""
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
                dconv = dconv,\
                dpc = 0.0,\
                path = path,\
                *args, **kwargs)
    # @profile
    def createDecoder(self, config, opt, *args, **kwargs):
        path = config["path"] if "path" in config else ""
        dconv = config["dconv"] if "dconv" in config else ""
        n_extra_layers = config["extra_layers"] if "extra_layers" in config else 0

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
                dconv = dconv,\
                channel = config['channel'],\
                bn = True,\
                dpc = 0.0,\
                n_extra_layers=n_extra_layers,\
                path = path,\
                *args, **kwargs)

    def createDCGAN_Dx(self, config_dcgan_dx, opt, *args, **kwargs):
        nc      = config_dcgan_dx["nc"] if "nc" in config_dcgan_dx else opt.nch
        wf      = json.loads(config_dcgan_dx["wf"].lower()) if "wf" in config_dcgan_dx else False
        dpc     = config_dcgan_dx["dpc"] if "dpc" in config_dcgan_dx else 0.25
        path    = config_dcgan_dx["path"] if "path" in config_dcgan_dx else ""
        bn      = json.loads(config_dcgan_dx["bn"].lower()) if "bn" in config_dcgan_dx else False
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
                                     grp=1,bn=bn,wf=wf, dpc=dpc,
                                     path=path,
                                     n_extra_layers=0)
        return dcgan_dx

    def createDCGAN_Dz(self, config_dcgan_dz, opt, *args, **kwargs):
        nz    = config_dcgan_dz["nz"]   if "nz" in config_dcgan_dz else opt.nzd
        wf  = json.loads(config_dcgan_dz["wf"].lower()) if "wf" in config_dcgan_dz else False
        dpc = config_dcgan_dz["dpc"] if "dpc" in config_dcgan_dz else 0.25
        path = config_dcgan_dz["path"] if "path" in config_dcgan_dz else ""
        bn      = json.loads(config_dcgan_dz["bn"].lower()) if "bn" in config_dcgan_dz else False
        dcgan_dz  = DCGAN_DzModelParallele.getDCGAN_DzByGPU(ngpu=opt.ngpu, nz=nz,\
                                     ncl=1024, fpd=0, n_extra_layers=0, dpc=dpc,
                                     nly = config_dcgan_dz['nlayers'],\
                                     ker = config_dcgan_dz['kernel'],\
                                     std = config_dcgan_dz['strides'],\
                                     pad = config_dcgan_dz['padding'],\
                                     act = config_dcgan_dz['act'],\
                                     limit = config_dcgan_dz['limit'],\
                                     channel = config_dcgan_dz['channel'],\
                                     dil= config_dcgan_dz["dilation"],\
                                     path = path,
                                     grp=1,bn=bn,wf=wf, bias = False)
        return dcgan_dz

    def createDCGAN_DXZ(self,config_dcgan_dxz, opt, *args, **kwargs):
        nc    = config_dcgan_dxz["nc"] if "nc" in config_dcgan_dxz else 1024
        wf  = json.loads(config_dcgan_dxz["wf"].lower()) if "wf" in config_dcgan_dxz else False
        dpc = config_dcgan_dxz["dpc"] if "dpc" in config_dcgan_dxz else 0.25
        path = config_dcgan_dxz["path"] if "path" in config_dcgan_dxz else ""
        bn      = json.loads(config_dcgan_dxz["bn"].lower()) if "bn" in config_dcgan_dxz else False
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
                                     path = path,
                                     dpc=dpc,wf=wf,bn = bn, bias = True, opt=None)
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
        # pdb.set_trace()
        name    = config["name"] if "name" in config else None
        path    = config["path"] if "path" in config else ""
        dconv   = config["dconv"] if "dconv" in config else ""
        dpc     = config["dpc"] if "dpc" in config else 0.0
        wf      = json.loads(config["wf"].lower()) if "wf" in config else False
        return EncoderDataParallele.getEncoder(name = name, ngpu = opt.ngpu,\
            dev     = opt.dev,\
            nz      = opt.nzd, nch = opt.nch,\
            ndf     = opt.ndf,\
            nly     = config['nlayers'],\
            ker     = config['kernel'],\
            dil     = config['dilation'],\
            std     = config['strides'],\
            pad     = config['padding'],\
            act     = config['act'],\
            limit   = config['limit'],\
            dconv   = dconv,\
            channel = config['channel'],\
            path    = path,
            config  = config,
            dpc     = dpc,\
            wf      = wf,\
            *args, **kwargs)

    def createDecoder(self, config, opt, *args, **kwargs):
        name        = config["name"] if "name" in config else None
        path        = config["path"] if "path" in config else ""
        dconv       = config["dconv"] if "dconv" in config else ""
        n_extra_layers= config["extra_layers"] if "extra_layers" in config else 0
        extra       = config["extra"] if "extra" in config else 0
        dpc         = config["dpc"] if "dpc" in config else 0.0
        # wf    = json.loads(config["wf"].lower()) if "wf" in config else False
        return DecoderDataParallele.getDecoder(name = name, ngpu = opt.ngpu,\
            nz      = opt.nzd, nch = opt.nch,\
            ndf     = opt.ndf,\
            nly     = config['nlayers'],\
            ker     = config['kernel'],\
            std     = config['strides'],\
            dil     = config['dilation'],\
            pad     = config['padding'],\
            opd     = config['outpads'],\
            act     = config['act'],\
            limit = config['limit'],\
            channel = config['channel'],\
            n_extra_layers=n_extra_layers,\
            bn      = True,
            path    = path,\
            dconv   = dconv,\
            dpc     = dpc,\
            extra   = extra,\
            *args, **kwargs)

    def createDCGAN_Dx(self, config_dcgan_dx, opt, *args, **kwargs):

        #another class could be called here if is not the generic class
        name    = config_dcgan_dx["name"] if "name" in config_dcgan_dx else None
        nc      = config_dcgan_dx["nc"] if "nc" in config_dcgan_dx else opt.nch
        wf      = json.loads(config_dcgan_dx["wf"].lower()) if "wf" in config_dcgan_dx else False
        dpc     = config_dcgan_dx["dpc"] if "dpc" in config_dcgan_dx else 0.25
        path    = config_dcgan_dx["path"] if "path" in config_dcgan_dx else ""
        extra   = config_dcgan_dx["extra"] if "extra" in config_dcgan_dx else 128
        bn      = json.loads(config_dcgan_dx["bn"].lower()) if "bn" in config_dcgan_dx else False
        prob    = json.loads(config_dcgan_dx["prob"].lower()) if "prob" in config_dcgan_dx else False
        #DCGAN_Dx class is called here
        return DCGAN_DxDataParallele.getDCGAN_DxDataParallele(name=name, ngpu = opt.ngpu,\
            isize   = 256, nc = nc,\
            ncl     = 512, ndf = opt.ndf, fpd=1,\
            nly     = config_dcgan_dx['nlayers'],\
            ker     = config_dcgan_dx['kernel'],\
            std     = config_dcgan_dx['strides'],\
            pad     = config_dcgan_dx['padding'],\
            act     = config_dcgan_dx['act'],\
            limit   = config_dcgan_dx['limit'],\
            dil     = config_dcgan_dx['dilation'],\
            channel = config_dcgan_dx['channel'],\
            path    = path,\
            prob    = prob,\
            extra   = extra,\
            grp     = 0,\
            bn      = bn,\
            wf      = wf,\
            dpc     = dpc,
            batch_size = opt.batchSize,\
            n_extra_layers=0)
         

    def createDCGAN_Dz(self, config_dcgan_dz, opt, *args, **kwargs):
        #DCGAN_DZ class is called here 
        name    = config_dcgan_dz["name"] if "name" in config_dcgan_dz else None
        nz      = config_dcgan_dz["nz"]   if "nz" in config_dcgan_dz else opt.nzd
        nc      = config_dcgan_dz["nc"]   if "nc" in config_dcgan_dz else opt.nzd
        wf      = json.loads(config_dcgan_dz["wf"].lower()) if "wf" in config_dcgan_dz else False
        dpc     = config_dcgan_dz["dpc"] if "dpc" in config_dcgan_dz else 0.25
        path    = config_dcgan_dz["path"] if "path" in config_dcgan_dz else ""
        extra   = config_dcgan_dz["extra"] if "extra" in config_dcgan_dz else 128
        bn      = json.loads(config_dcgan_dz["bn"].lower()) if "bn" in config_dcgan_dz else False
        prob    = json.loads(config_dcgan_dz["prob"].lower()) if "prob" in config_dcgan_dz else False
        return DCGAN_DzDataParallele.getDCGAN_DzDataParallele( name =  name, ngpu = opt.ngpu,\
            nc      = nc, nz = nz,\
            ncl     = 512, ndf=opt.ndf, fpd=0,\
            nly     = config_dcgan_dz['nlayers'],\
            ker     = config_dcgan_dz['kernel'],\
            std     = config_dcgan_dz['strides'],\
            pad     = config_dcgan_dz['padding'],\
            act     = config_dcgan_dz['act'],\
            limit = config_dcgan_dz['limit'],\
            dil     = config_dcgan_dz['dilation'],\
            channel = config_dcgan_dz['channel'],\
            grp     =0, bn=bn,wf=wf, dpc=dpc,
            path    = path,\
            prob    = prob,\
            extra   = extra,
            batch_size = opt.batchSize,\
            n_extra_layers=0)

    def createDCGAN_DXZ(self, config_dcgan_dxz,opt, *args, **kwargs):
        #DCGAN_DXZ class is called here
        name  = config_dcgan_dxz["name"] if "name" in config_dcgan_dxz else None
        nc    = config_dcgan_dxz["nc"] if "nc" in config_dcgan_dxz else 1024
        wf    = json.loads(config_dcgan_dxz["wf"].lower()) if "wf" in config_dcgan_dxz else False
        prob  = json.loads(config_dcgan_dxz["prob"].lower()) if "prob" in config_dcgan_dxz else False
        bias  = json.loads(config_dcgan_dxz["bias"].lower()) if "bias" in config_dcgan_dxz else False
        dpc   = config_dcgan_dxz["dpc"] if "dpc" in config_dcgan_dxz else 0.25
        path  = config_dcgan_dxz["path"] if "path" in config_dcgan_dxz else ""
        extra = config_dcgan_dxz["extra"] if "extra" in config_dcgan_dxz else 128
        bn    = json.loads(config_dcgan_dxz["bn"].lower()) if "bn" in config_dcgan_dxz else False
        return DCGAN_DXZDataParallele.getDCGAN_DXZDataParallele(name = name, ngpu=opt.ngpu, nc=nc, 
            nly     = config_dcgan_dxz['nlayers'],\
            ker     = config_dcgan_dxz['kernel'],\
            std     = config_dcgan_dxz['strides'],\
            pad     = config_dcgan_dxz['padding'],\
            act     = config_dcgan_dxz['act'],\
            limit   = config_dcgan_dxz['limit'],\
            dil     = config_dcgan_dxz['dilation'],\
            channel = config_dcgan_dxz['channel'],\
            grp     = 0, bn=bn,wf=wf, dpc=dpc,\
            path    = path,\
            prob    = prob,\
            bias    = bias,\
            extra   = extra,\
            n_extra_layers=0,batch_size = opt.batchSize)

    
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


