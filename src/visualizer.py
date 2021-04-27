u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
# from profile_support import profile
from common_nn import *
from common_torch import * 
import plot_tools as plt
from generate_noise import latent_resampling, noise_generator
from generate_noise import lowpass_biquad
from database_sae import random_split 
from leave_p_out import k_folds
from ex_common_setup import dataset2loader
from database_sae import thsTensorData
import json
from pytorch_summary import summary
import pdb
from conv_factory import *
import GPUtil
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import pdb
import matplotlib.pyplot as plt
from scipy import signal


class visualizer:

    def __init__(self,cv):

        """
        Args
        cv  [object] :  content all parsing paramaters from the flag when lauch the python instructions
        """
        super(visualizer, self).__init__()
    
        self.cv = cv
        self.gr_norm = []

        # define as global variable the cv object. 
        # And therefore this latter is become accessible to the methods in this class
        globals().update(cv)
        # define as global opt and passing it as a dictonnary here
        globals().update(opt.__dict__)
        no_of_layers=0
        conv_layers=[]
        self.dp_mode =  True
        cpus  =  int(os.environ.get('SLURM_NPROCS'))
        #we determine in which kind of environnement we are 
        if(cpus ==1 and opt.ngpu >=1):
            print('ModelParallele to be builded ...')
            self.dp_mode = False
            factory = ModelParalleleFactory()
        elif(cpus >1 and opt.ngpu >=1):
            print('DataParallele to be builded ...')
            factory = DataParalleleFactory()
            self.dp_mode = True
        else:
            print('environ not found')
        net = Network(factory)

        # pdb.set_trace()
        self.Fef = net.Encoder(opt.config['encoder'], opt)
        self.Gdf = net.Decoder(opt.config['decoder'], opt)
        _Fef_path = './network/bb_ls64_nf8_nzd32/Fef_1000.pth'
        _Gdf_path = './network/bb_ls64_nf8_nzd32/Gdf_1000.pth'
            # if self.strategy['tract']['filtered']:
            #     if None in n:        
            #         self.FGf = [self.Fef,self.Gdf]
            #         self.oGfxz = reset_net(self.FGf,func=set_weights,lr=glr,b1=b1,b2=b2,
            #                 weight_decay=0.00001)
            #     else:
            #         print("Filtered generators: {0} - {1}".format(*n))
        self.Fef.load_state_dict(tload(_Fef_path)['model_state_dict'])
        self.Gdf.load_state_dict(tload(_Gdf_path)['model_state_dict'])    
        # self.oGfxz = Adam(ittc(self.Fef.parameters(),self.Gdf.parameters()),
        #                   lr=glr,betas=(b1,b2),weight_decay=0.00001)

        model_children = list(self.Fef.children())
        pdb.set_trace()

        for child in model_children:
          if type(child)==nn.Conv1d:
            no_of_layers+=1
            conv_layers.append(child)
          elif type(child)==nn.Sequential:
            for layer in child.children():
              if type(layer)==nn.Conv1d:
                no_of_layers+=1
                conv_layers.append(layer)

        print(no_of_layers)
        # pdb.set_trace()
        # X = []

        for _,batch in enumerate(trn_loader):
            # Load batch
            # pdb.set_trace()

            place = opt.dev if self.dp_mode else ngpu-1

            xd_data,_,zd_data,_,_,_,_ = batch
            Xd = Variable(xd_data).to(place)# BB-signal
            zd = Variable(zd_data).to(place)

            wnx,_,_ = noise_generator(Xd.shape,zd.shape,device,rndm_args)
            
            wnx = wnx.to(Xd.device)
            # wnz = wnz.to(zd.device)
            # 1. Concatenate inputs
            X_inp = zcat(Xd,wnx)
            # X.append(X_inp)
            break

            # z_inp = zcat(zd,wnz)

        #passing the values in the convnets layers
        #layers #1 : 
        results = [conv_layers[0](X_inp[0,0,:])]
        #from #2 til the end
        for i in range(1, len(conv_layers)):
            results.append(conv_layers[i](results[-1]))
        outputs = results

        # pdb.set_trace()
        fs = 100 #Hz

        for num_layer in range(len(outputs)):
            layer_viz = outputs[num_layer][0, :, :]
            layer_viz = layer_viz.data
            print("Layer ",num_layer+1)
            for i, filter in enumerate(layer_viz):
                if i == 16: 
                    break
                # pdb.set_trace()
                x  = filter.cpu()
                # nx = len(x)
                f, t, Sxx = signal.spectrogram(x, fs)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                # plt.plot(range(len(x)), x)
                # plt.imshow(filter, cmap='gray')
                # plt.axis("off")
                print("savefig results %s of layer %s ..."%(i, num_layer))
                plt.savefig(os.path.join(outf,"filter_%s_%s.png"%(i,num_layer)),\
                bbox_inches='tight',dpi = 300)

            # plt.show()
            plt.close()

        # z = torch.cat(z)
        # torch.save(z, './database/tweaked/data/latent_sampling.pth')
        # print('saved ./database/tweaked/data/latent_sampling.pth ')



