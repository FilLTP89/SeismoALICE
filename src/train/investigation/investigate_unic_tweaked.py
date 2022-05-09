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
# import torch.nn as nn 
# from torch.utils.tensorboard import SummaryWriter
from factory.conv_factory import *
from database.toyset import Toyset, get_dataset
from configuration import app
# from torch.nn import DataParallel as DP
import numpy as np
from scipy.stats import norm

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
        self.vld_loader_STEAD, self.vtm_STEAD = vld_loader, vtm
        self.outf = outf

        # Niigata dataset
        # hdf5 = Hdf5Dataset(hdf5_file="dataset_niigata.hdf5",
        #     root_dir="../Niigata/"
        # )
        # vld_set, _= torch.utils.data.random_split(hdf5, [800, 11])
        # self.vld_loader_Niigata = DataLoader(dataset=vld_set, 
        #     batch_size=128,shuffle=True
        # )

        self.opt = opt
        # read json file
        try:
            with open('./config/investigation.json') as json_file:
                network_file = json.load(json_file)
        except OSError:
                app.logger.info("|file {}.json not found".format(opt.config))
                network_file =  None

        network_path = network_file["network.path"]        

        # Initialization of the neural networks
        net = Network(DataParalleleFactory())

        self.Fyx = net.Encoder(self.opt.config['F' ], self.opt)
        self.Gy  = net.Decoder(self.opt.config['Gy'], self.opt)

        self.Dy  = net.DCGAN_Dx( self.opt.config['Dy' ], self.opt)
        # self.Dsy = net.DCGAN_DXZ(self.opt.config['Dsy'], self.opt)
        breakpoint()
        # loading saved weight of the network
        app.logger.info(f'loading data from {network_path["encoder"]} ...')
        self.Fyx.load_state_dict(tload(network_path["encoder"])['model_state_dict'])

        app.logger.info(f'loading data from {network_path["decoder"]} ...')
        self.Gy.load_state_dict(tload(network_path["decoder"])['model_state_dict'])

        app.logger.info(f'loading data from {network_path["Dy"]} ...')
        self.Dy.load_state_dict(tload(network_path["Dy"])['model_state_dict'])

        # app.logger.info(f'loading data from {network_path["Dsy"]} ...')
        # self.Dsy.load_state_dict(tload(network_path["Dsy"])['model_state_dict'])

        self.FGf = [self.Fyx, self.Gy,self.Dy, self.Dsy]
        # breakpoint()
        #printing the number of parameters of the encoder/decoder
        print(" Parameters of  Decoders/Decoders ")
        count_parameters(self.FGf)

        Xf,Xr,Xd = self.calculation_of_filtered_broadband_signal()
        self.plot_latent_space_distribution()
        # self.plot_signal(Xf,Xr,Xd,self.opt.outf)
    
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

    def plot_distribution(self,zlf, zhf):
        fig, axes = plt.subplots(figsize=(8, 5), sharey=True)
        ax = sns.histplot({"zyy":zhf.reshape(-1), "zxx":zlf.reshape(-1)},kde=False,
                stat="density", common_norm=True, element="poly",fill=False)
        ax.set_xlim(-10,10)
        fig = ax.get_figure()
        fig.show()
        fig.savefig("file_zxx.png")

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
    
    def get_marginal_y(self):
        y,_ = self.fetch_data()
        y   = y.to(app.DEVICE)

        wny,*others = noise_generator(y.shape,y.shape,app.DEVICE,{'mean':0., 'std':self.std})
        fty         = self.Dy(zcat(y,wny))
        wny,*others = noise_generator(fty.shape,fty.shape,app.DEVICE,{'mean':0., 'std':self.std})       
        Dreal       = self.Dsy(zcat(fty,wny))

        nch, nz = 8,128
        zd = torch.empty([y.shape[0],nch,nz]).normal_(**{'mean':0., 'std':0.763}).to(app.DEVICE)
        yr = self.Gy(zd)

        wny,*others = noise_generator(y.shape,y.shape,app.DEVICE,{'mean':0., 'std':self.std})
        fty         = self.Dy(zcat(yr,wny))
        wny,*others = noise_generator(fty.shape,fty.shape,app.DEVICE,{'mean':0., 'std':self.std})       
        Dfake       = self.Dsy(zcat(fty,wny))

        return Dreal, Dfake
    
    def plot_latent_space_distribution(self):
        app.logger.info('plotting latent space distribution ...')
        Xd, Xf = self.fetch_data()
        Xf = Xf.to(app.DEVICE)
        Xd = Xd.to(app.DEVICE)
        # Extraction of low frequency signal
        # breakpoint()
        self.Fyx.eval()
        self.Gy.eval()
        wnx,*others = noise_generator(Xf.shape,Xf.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Xf_inp = zcat(Xf,wnx)
        zHF_gen ,zLF_gen = self.Fyx(Xf_inp)

        nch, nz = 4,128
        gauss = torch.empty([Xf.shape[0],nch,nz]).normal_(**{'mean':0., 'std':1.}).to(app.DEVICE)

        # from torch to numpy
        gauss   = gauss.reshape(-1).cpu().data.numpy().copy()
        zhf     = zHF_gen.reshape(-1).cpu().data.numpy().copy()
        zlf     = zLF_gen.reshape(-1).cpu().data.numpy().copy()

        mu, std = norm.fit(gauss)

        # plt.hist(gauss, bins=25, density=True, alpha=0.6, color='g')
        plt.hist(zhf, bins=25, density=True, alpha=0.6, color='r')
        xmin, xmax = plt.xlim()

        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        app.logger.info('saving latent space distribution ...')
        plt.savefig(os.path.join(self.outf, "latent_distribution.png"))

    def calculation_of_filtered_broadband_signal(self):
        # with torch.no_grad():
        Xd, Xf = self.fetch_data()
        Xf = Xf.to(app.DEVICE)
        Xd = Xd.to(app.DEVICE)
        # Extraction of low frequency signal
        # breakpoint()
        self.Fyx.eval()
        self.Gy.eval()
        wnx,*others = noise_generator(Xf.shape,Xf.shape,app.DEVICE,{'mean':0., 'std':self.std})
        Xf_inp = zcat(Xf,wnx)
        _ ,_zLF_gen = self.Fyx(Xf_inp)
        # X_gen           = self.Gy(zLF_gen,zHF_gen)
        # _,zLF_rec       = self.Fyx(zcat(X_gen,wnx))
       
        # zHF,_ = self.Fyx(zcat(Xd,wnx))
        

        nch, nz = 4,128
        zyy = torch.empty([Xf.shape[0],nch,nz]).normal_(**{'mean':0., 'std':1.}).to(app.DEVICE)
        zxy = torch.empty([Xf.shape[0],nch,nz]).normal_(**{'mean':0., 'std':1.}).to(app.DEVICE)
        # wnx,*others = noise_generator(Xf.shape,Xf.shape,app.DEVICE,{'mean':0., 'std':self.std})
        # _wn = wn*0.7
        # temp_x = self.Gy(zxy,zyy)
        
        # _Xd = self.Gy(o0l(zLF_gen),zyy)
        wnx = torch.empty(Xf.shape).normal_(**{'mean':0., 'std':1.}).to(app.DEVICE)
        zLF_gen, _ = self.Fyx(zcat(Xd,wnx))
        # import numpy as np
        # import matplotlib.pyplot as plt
        # s = np.random.standard_cauchy(128*4*128)
        # s = s.reshape(128,4,128)
        # wn = torch.from_numpy(s)
        Xr = self.Gy(zLF_gen,o0l(zyy))
        # self.plot_distribution(wnx.data.cpu().numpy().copy(),zHF.data.cpu().numpy().copy())
        return Xf, Xr, Xd 

    def plot_box_plot(self):
        Dreal, Dfake = self.get_marginal_y()
        Dreal = Dreal.data.cpu().numpy().copy()
        Dfake = Dfake.data.cpu().numpy().copy()

        data = [Dreal,Dfake]
        fig = plt.figure(figsize =(10, 7))
 
        # Creating axes instance
        ax = fig.add_axes([0, 0, 1, 1])
        
        # Creating plot
        bp = ax.boxplot(data)
        
        # show plot
        plt.savefig(os.path.join(self.opt.outf,"box_plot_marginal_y.png"))
        plt.close()
        app.logger.info("saving box plot for marginal y")

    def plot_signal(self,Xf, Xr,Xd, outf,*args,**kwargs):
        cnt = 0
        clr = ['black', 'blue','red', 'orange']
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
            
            hax0.plot(vtm,gt,color='red',label=r'$G(F(\mathbf{x},N(0,I))$',linewidth=1.2)
            # hax0.plot(vtm,bt,color='blue',label=r'$\mathbf{y}$',linewidth=1.2, alpha=0.70)
            hax0.plot(vtm,ot,color='black',label=r'$\mathbf{x}$',linewidth=1.2, alpha=0.70)

            hax1.loglog(vfr,of,color='black',label=r'$\mathbf{x}$',linewidth=2)
            # hax1.loglog(vfr,bf,color='blue',label=r'$\mathbf{y}$',linewidth=2)
            hax1.loglog(vfr,gf,color='red',label=r'$G(F(\mathbf{x}),F(\mathbf{y}))$',linewidth=2)
            
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







        
        
