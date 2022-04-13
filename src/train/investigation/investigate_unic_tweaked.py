# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Train and Test AAE'''
u'''Required modules'''

from copy import deepcopy
from tkinter import W
import matplotlib.pyplot as plt
from common.common_nn import *
from common.common_torch import * 
# import plot.plot_tools as plt
import plot.investigation as ivs
import seaborn as sns
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

        self.std = 0.763
        self.strategy = strategy

        # STEAD dataset 
        self.vld_loader_STEAD, self.vtm_STEAD = vld_loader, vtm

        # Niigata dataset
        # hdf5 = Hdf5Dataset(hdf5_file="dataset_niigata.hdf5",
        #     root_dir="../Niigata/"
        # )
        # vld_set, _= torch.utils.data.random_split(hdf5, [800, 11])
        # self.vld_loader_Niigata = DataLoader(dataset=vld_set, 
        #     batch_size=128,shuffle=True
        # )

        self.opt = opt

        # Initialization of the neural networks
        net = Network(DataParalleleFactory())

        n = self.strategy['unique']

        self.Fyx = net.Encoder(self.opt.config['F'],  self.opt)
        self.Gy  = net.Decoder(self.opt.config['Gy'], self.opt)

        self.Fyx  = nn.DataParallel(self.Fyx).cuda()
        self.Gy  = nn.DataParallel(self.Gy).cuda()

        # loading saved weight of the network

        app.logger.info(f'loading data from {n[0]} ...')
        app.logger.info(f'loading data from {n[1]} ...')
        self.Fyx.load_state_dict(tload(n[0])['model_state_dict'])
        self.Gy.load_state_dict(tload(n[1])['model_state_dict'])
        self.FGf = [self.Fyx, self.Gy]
        # breakpoint()
        #printing the number of parameters of the encoder/decoder
        print(" Parameters of  Decoders/Decoders ")
        count_parameters(self.FGf)

        Xf, Xr,Xd = self.calculation_of_filtered_broadband_signal()
        self.plot_signal(Xf,Xr,Xd,self.opt.outf)
    
    # @profile            
    # def investigate_signal(self, plotting=False):
    #     """
    #         In this function we try to calculate the gofs for the dataset.
    #         Furthermore, we print plots signals and of for the training
    #     """
    #     app.logger.info("generating result...")
        
    #     # For the STEAD validation dataset 
    #     outfile = self.opt.outf+"STEAD"
    #     EG_STEAD, PG_STEAD  = plt.get_gofs(tag ='broadband', 
    #         Qec = self.Fyx, Pdc = self.Gy, trn_set=self.vld_loader_STEAD, 
    #         pfx="unique_broadband_STEAD_investigate", opt = self.opt, std=self.std, 
    #         outf = opt.outf
    #     )
    #     # values have been saved for post processing.
    #     np.save(outfile+"_EG_STEAD_filtered.npy",EG_STEAD)
    #     np.save(outfile+"_PG_STEAD_filtered.npy",PG_STEAD)
    #     plt.plot_eg_pg(EG_STEAD, PG_STEAD, outf = str(opt.outf+"STEAD"), pfx='broadband')

    #     if plotting:
    #         ivs.plot_signal_and_reconstruction(
    #             vld_set = self.vld_loader_STEAD,
    #             encoder = self.Fyx,
    #             decoder = self.Gy, 
    #             outf = outfile, 
    #             opt = opt, 
    #             device='cuda')
    #         app.logger.info(" plottings EG_STEAD, PG_STEAD points is finished")
        
    #     # For the Niigata dataset 
    #     outfile = self.opt.outf+"Niigata"
    #     EG_Niigata, PG_Niigata  = plt.get_gofs(tag ='broadband', 
    #         Qec = self.Fyx, Pdc = self.Gy, trn_set=self.vld_loader_Niigata, 
    #         pfx="unique_broadband_Niigata_investigate", opt = self.opt, std=self.std, 
    #         outf = opt.outf
    #     )
    #     np.save(outfile+"_EG_Niigata.npy",EG_Niigata)
    #     np.save(outfile+"_PG_Niigata.npy",PG_Niigata)        

    #     plt.plot_eg_pg(EG_Niigata, PG_Niigata, outf = str(opt.outf+"Niigata"), pfx='broadband')
    #     app.logger.info(" plottings PG_Niigata, PG_Niigata points is finished")
   
    # def investigate_latent_variable(self):
    #     app.logger.info(" Generating latent signals ...")
    #     outfile = self.opt.outf+"STEAD/latent/broadband"
    #     app.logger.info(f"File should be saved at directory : {outfile} ")
        
    #     ivs.plot_latent_signals(
    #             vld_set = self.vld_loader_STEAD,
    #             encoder = self.Fyx, 
    #             outf = outfile, 
    #             opt = opt, 
    #             device='cuda')

    def fetch_data(self):
        batch = next(iter(self.vld_loader_STEAD))
        xt_data, xf_data,*others = batch
        if torch.isnan(torch.max(xt_data)) or torch.isnan(torch.max(xt_data)):
                app.logger.debug("your model contain nan value "
                    "this signals will be withdrawn from the training "
                    "but style be present in the dataset. \n"
                    "Then, remember to correct your dataset")
                mask   = [not torch.isnan(torch.max(xt_data[e,:])).tolist() for e in range(len(xt_data))]
                index  = np.array(range(len(xt_data)))
                xt_data.data = xt_data[index[mask]]
                xf_data.data = xf_data[index[mask]]
        return xt_data, xf_data
    

    def calculation_of_filtered_broadband_signal(self):
        with torch.no_grad():
            Xd, Xf = self.fetch_data()
            Xf = Xf.to(app.DEVICE)
            Xd = Xd.to(app.DEVICE)
            # Extraction of low frequency signal
            wnx,*others = noise_generator(Xf.shape,Xf.shape,app.DEVICE,{'mean':0., 'std':self.std})
            Xf_inp = zcat(Xf,wnx)
            _, zLF = self.Fyx(Xf_inp)
            # zHLb,zLHb = self.Fyx(zcat(Xd,wnx))
            nch, nz = 4,128
            wn = torch.empty([Xf.shape[0],nch,nz]).normal_(**app.RNDM_ARGS).to(app.DEVICE)
            Xr = self.Gy(zLF,wn)
        return Xf, Xr, Xd 


    def plot_signal(self,Xf, Xr,Xd, outf,*args,**kwargs):
        cnt = 0
        clr = ['black', 'blue','red', 'red']
        sns.set(style="whitegrid")
        pfx = "from_filtered_to_broadband"
        vtm = self.vtm_STEAD
    
        Xf_fsa = tfft(Xf,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xr_fsa = tfft(Xr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xd_fsa = tfft(Xd,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xf  = Xf.cpu().data.numpy().copy()
        Xr  = Xr.cpu().data.numpy().copy()
        Xd  = Xd.cpu().data.numpy().copy()
        vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)

        for (io, ig) in zip(range(Xf.shape[0]),range(Xr.shape[0])):
            cnt +=1
            ot,gt,bt = Xf[io, 1, :]  ,Xr[ig, 1, :], Xd[ig,1,:]
            of,gf,bf = Xf_fsa[ig,1,:],Xr_fsa[io,1,:],Xd_fsa[ig,1,:]
            _,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
            
            hax0.plot(vtm,gt,color=clr[3],label=r'$G(F(\mathbf{x},N(0,I))$',linewidth=1.2)
            hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{x}$',linewidth=1.2, alpha=0.70)
            hax0.plot(vtm,bt,color=clr[2],label=r'$\mathbf{y}$',linewidth=1.2, alpha=0.70)

            hax1.loglog(vfr,of,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
            hax1.loglog(vfr,gf,color=clr[3],label=r'$G(F(\mathbf{x},N(0,I))$',linewidth=2)
            hax1.loglog(vfr,bf,color=clr[2],label=r'$G(F(\mathbf{y})$',linewidth=2)
            
            hax0.set_xlim(0.0,int(vtm[-1]))
            hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax0.set_ylim(-1.0,1.0)
            hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax0.set_title('ALICE',fontsize=20,fontweight='bold')
            hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            hax0.legend(loc = "lower right",frameon=False)
            hax1.legend(loc = "lower right",frameon=False)
        
            plt.savefig(os.path.join(outf,"res_bb_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                            bbox_inches='tight',dpi = 500)
            plt.close()
            app.logger.info("saving %sres_bb_aae_%s_%u_%u ... "%(outf,pfx,cnt,io))







        
        
