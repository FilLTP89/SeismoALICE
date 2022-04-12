# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from copy import deepcopy
from common.common_nn import *
from common.common_torch import * 
import plot.plot_tools as plt
import plot.investigation as ivs
import profiling.profile_support as profile
from tools.generate_noise import latent_resampling, noise_generator
from database.database_sae import random_split 
from database.database_sae import thsTensorData
from database.custom_dataset import Hdf5Dataset
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
from database.toyset import Toyset, get_dataset
from configuration import app
from tqdm import  tqdm,trange
import numpy as np

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

# coder en dure dans le programme 
class investigator(object):
    """
        This class have de responsability to invesitigate the neural networks.
    """
    def __init__(self,cv):
        super(investigator, self).__init__()
        globals().update(cv)
        globals().update(opt.__dict__)
        """
            Setting up here the dataset and AutoEncoders needed for the trainigs. 
        """

        self.std = 1.0
        self.strategy = strategy

        # STEAD dataset 
        self.vld_loader_STEAD = vld_loader
        summary_dir = './runs_both/broadband/zyy16/back-test/nsy1280/test-dxy/test-5/'
        self.writer_latent = SummaryWriter(f'{summary_dir}/investigation/')
        self.writer_latent_img = SummaryWriter(f'{summary_dir}/investigation/images/')

        self.opt = opt

        # Initialization of the neural networks
        net = Network(DataParalleleFactory())

        n = self.strategy['unique']

        self.F_  = net.Encoder(opt.config['F'],  self.opt)
        self.Gy  = net.Decoder(opt.config['Gy'], self.opt)

        self.F_  = nn.DataParallel(self.F_).cuda()
        self.Gy  = nn.DataParallel(self.Gy).cuda()
        self.Dzxy = 

        # loading saved weight of the network
        self.F_.load_state_dict(tload(n[0])['model_state_dict'])
        self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
        self.FGf = [self.F_, self.Gy]

        #printing the number of parameters of the encoder/decoder
        print(" Parameters of  Decoders/Decoders ")
        count_parameters(self.FGf)
        app.logger.info(f" Summary directory    : {summary_dir}")
        app.logger.info(f" Loading Encoder from :  {n[0]}")
        app.logger.info(f" Loading Decoder from :  {n[1]}")

    def generate_latent_variable(self, batch, nch_zd,nzd, nch_zf = 128,nzf = 128):
        zyy  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        zxx  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)

        zyx  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        zxy  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=self.std).to(app.DEVICE, non_blocking = True)
        return zyy, zyx, zxx, zxy
    
    def investigate_latent_variable(self):
        app.logger.info(" Investigation of latent signals ...")
        torch.random.manual_seed(0)
        # self._fake_filtered_signals()
        self._extract_latent_space()
      

    def _extract_latent_space(self):
        bar = tqdm(self.vld_loader_STEAD)

        nch_zd, nzd = 16,128
        nch_zf, nzf =  8,128

        _zxy = []
        _zyx = []
        _zxx = []
        _zyy = []
        
        for b,batch in enumerate(bar):
            y,x, *others = batch
            y = y.to(app.DEVICE, non_blocking = True)
            x = x.to(app.DEVICE, non_blocking = True)

            zyy,zyx,*other=self.generate_latent_variable(
                    batch=len(y),nzd=nzd,nch_zd=nch_zd,
                    nzf=nzf,nch_zf=nch_zf
                )
            wnx,*others = noise_generator(x.shape,zyy.shape,
                    app.DEVICE,{'mean':0., 'std':self.std}
                )

            x_inp   = zcat(x,wnx)
            zxx_F, zxy_F, *others = self.F_(x_inp)
            zxy_gen = zcat(zxy_F)

            _zxy.append(zxy_gen.cpu().data.numpy().copy())
            _zxx.append(zxx_F.cpu().data.numpy().copy())

            wny,*others = noise_generator(y.shape,zyy.shape,
                    app.DEVICE,{'mean':0., 'std':self.std}
                )
            y_inp   = zcat(y,wny)
            zyy_F,zyx_F,*other = self.F_(y_inp) 
            zyx_gen = zcat(zyx_F)
            _zyx.append(zyx_gen.cpu().data.numpy().copy())
            _zyy.append(zyy_F.cpu().data.numpy().copy())

        np.save(outf+"zxy.npy", _zxy)
        np.save(outf+"zyx.npy", _zyx)
        np.save(outf+"zxx.npy", _zxx)
        np.save(outf+"zyy.npy", _zyy)
        app.logger.info(f"latent space saved in {outf} ...")

            # #extraction of commmon part variables
            # bar.set_postfix(status='extracting zxy - zyx part ...')
            # for idx in range(opt.batchSize//torch.cuda.device_count()):
            #     self.writer_latent.add_histogram("common/zyx", zyx_gen[idx,:],b)
            # for idx in range(opt.batchSize//torch.cuda.device_count()):
            #     self.writer_latent.add_histogram("common/zxy", zxy_gen[idx,:],b)

            # #extraction specific pars
            # bar.set_postfix(status='extracting zyy - zxx part ...')
            # for idx in range(opt.batchSize//torch.cuda.device_count()):
            #     self.writer_latent.add_histogram("broadband/zxx", zxx_F[idx,:],b)
            # for idx in range(opt.batchSize//torch.cuda.device_count()):
            #     self.writer_latent.add_histogram("broadband/zyy", zyy_F[idx,:],b)

    def _fake_filtered_signals(self):
        app.logger.info("Extracting images")
        
        #extracting values 
        # figure_fl, gof_fl = plt.plot_generate_classic(
        #         tag='broadband',Qec=deepcopy(self.F_),Pdc=deepcopy(self.Gy),
        #         trn_set=self.vld_loader_STEAD,pfx="vld_set_bb_unique_hack",
        #         opt=opt,outf=outf, save=False
        #     )
        # self.writer_latent_img.add_figure('Fake zy=0 Filtered',figure_fl, 1)
        # self.writer_latent_img.add_figure('Fake zy=0 Goodness of Fit Filtered',gof_fl, 1)
        # app.logger.info("Filtered Images dones ...")

        figure_hb, gof_hb =  plt.plot_generate_classic(
                tag='broadband',Qec=deepcopy(self.F_),Pdc=deepcopy(self.Gy),
                trn_set=self.vld_loader_STEAD,pfx="vld_set_bb_unique_hack",
                opt=opt,outf=outf, save=False
        )
        self.writer_latent_img.add_figure('Hybrid signals',figure_hb, 2)
        self.writer_latent_img.add_figure('Goodness of Fit Hybrid',gof_hb, 2)
        app.logger.info("Hybrid Images dones ...")




