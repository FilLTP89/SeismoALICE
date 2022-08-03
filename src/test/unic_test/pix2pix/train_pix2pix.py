import torch
from app.trainer.unic.unic_trainer import UnicTrainer
from common.common_nn import zerograd,zcat,modalite
from plot.plot_tools import get_gofs, plot_generate_classic
from tools.generate_noise import noise_generator
from common.common_nn import get_accuracy
from common.common_torch import *
from configuration import app
from test.unic_test.pix2pix.strategy_discriminator_pix2pix import StrategyDiscriminatorPix2Pix
from test.unic_test.pix2pix.strategy_generator_pix2pix import StrategyGeneratorPix2Pix

class Pix2Pix(UnicTrainer):
    def __init__(self,cv, trial=None):
        losses_disc = {
            'epochs':'',                'modality':'',
            'Dloss':'',                 'Dloss_xy':'' 
        }

        losses_gens = {
            'epochs':'',                'modality':'',
            'Gloss':'',                 'Gloss_pix2pix':'',         
            'Gloss_xy':'',              'Gloss_rec':'',
            'Gloss_rec_y':''
        }

        prob_disc = {
            'epochs':'',            'modality':'',
            'Dreal_xy':'',          'Dfake_xy':'',
        }

        gradients_gens = {
            'epochs':'',    'modality':'',
            'Fxy':'',
        }

        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dxy':'',
        }

        super(Pix2Pix, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens, prob_disc  = prob_disc,
        strategy_discriminator = StrategyDiscriminatorPix2Pix, 
        strategy_generator = StrategyGeneratorPix2Pix, 
        gradients_gens = gradients_gens, 
        gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1. Pix2Pix
            wnx,*others = noise_generator(x.shape,y.shape,app.DEVICE,app.NOISE)
            x_inp = zcat(x,wnx)
            y_gen = self.gen_agent.Fxy(x_inp)

            Dreal_xy, Dfake_xy = self.disc_agent.discriminate_conjointe_xy(x,y,y_gen)
            Dloss_xy= self.bce_logit_loss(Dreal_xy.reshape(-1),o1l(Dfake_xy.reshape(-1))) +\
                            self.bce_logit_loss(Dfake_xy.reshape(-1),o0l(Dfake_xy.reshape(-1)))
            
            # 2. Summation of losses
            Dloss_pix2pix      = Dloss_xy
            Dloss              = Dloss_pix2pix/2
            
            if modality == 'train':
                zerograd([self.gen_agent.optimizer_encoder, self.disc_agent.optimizer])
                Dloss.backward()
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
            
            self.losses_disc['epochs'   ] = epoch
            self.losses_disc['modality' ] = modality
            self.losses_disc['Dloss'    ] = Dloss.tolist()
            self.losses_disc['Dloss_xy' ] = Dloss_xy.tolist()
            self.losses_disc['Dloss_pix2pix'] = Dloss_pix2pix.tolist()
            self.losses_disc_tracker.update()

            self.prob_disc['epochs'  ] = epoch
            self.prob_disc['modality'] = modality
            self.prob_disc['Dreal_xy'] = torch.sigmoid(Dreal_xy).mean().tolist()
            self.prob_disc['Dfake_xy'] = torch.sigmoid(Dfake_xy).mean().tolist()
            self.prob_disc_tracker.update()
            
    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder,self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1. Pix2Pix
            wnx,*others = noise_generator(x.shape,y.shape,app.DEVICE,app.NOISE)
            x_inp = zcat(x,wnx)
            y_gen      = self.gen_agent.Fxy(x_inp)
            _, Dfake_xy = self.disc_agent.discriminate_conjointe_xy(x,y,y_gen)
            Gloss_xy    = self.bce_logit_loss(Dfake_xy.reshape(-1),o1l(Dfake_xy.reshape(-1)))
            
            # 2. Reconstruction
            Gloss_rec_y = self.l1_loss(y,y_gen)*app.LAMBDA_IDENTITY
            
            # 3. Summation of losses
            Gloss_pix2pix   = Gloss_xy
            Gloss_rec       = Gloss_rec_y
            Gloss           = Gloss_pix2pix + Gloss_rec

            if modality == 'train':
                zerograd([self.gen_agent.optimizer_encoder,self.disc_agent.optimizer])
                Gloss.backward()
                self.gen_agent.optimizer_encoder.step()
                self.gen_agent.track_gradient(epoch)
            
            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss'        ] = Gloss.tolist()
            self.losses_gens['Gloss_pix2pix'] = Gloss_pix2pix.tolist()
            self.losses_gens['Gloss_rec'    ] = Gloss_rec.tolist()
            self.losses_gens['Gloss_xy'     ] = Gloss_xy.tolist()
            self.losses_gens['Gloss_rec_y'  ] = Gloss_rec_y.tolist()
            self.losses_gen_tracker.update()

    def on_test_epoch(self, epoch, bar):
        # method is overwritten
        with torch.no_grad(): 
            torch.manual_seed(100)
            if epoch%self.opt.config["hparams"]['test_epochs'] == 0:
                self.validation_writer.set_step(mode='test', step=epoch)
                bar.set_postfix(status='saving accuracy and images ... ')
                accuracy_hb = get_accuracy(tag='pix2pix',plot_function=get_gofs,
                    encoder = self.gen_agent.Fxy,
                    decoder = None,
                    vld_loader = self.data_tst_loader,
                    pfx ="vld_set_bb_unique",opt= self.opt,
                    outf = self.opt.outf, save = False
                )
                bar.set_postfix(accuracy_hb=accuracy_hb)
                self.validation_writer.add_scalar('Accuracy/Filtered2Broadband',accuracy_hb,epoch)
                
                # plot hybrid filtred reconstruction signal and gof
                figure_hf, gof_hf = plot_generate_classic(tag ='pix2pix',
                    Qec= self.gen_agent.Fxy, Pdc= None,
                    trn_set=self.data_tst_loader, pfx="vld_set_bb_unique",
                    opt=self.opt, outf= self.opt.outf, save=False)
                bar.set_postfix(status='saving images STFD/GOF Pix2Pix ...')
                self.validation_writer.add_figure('STFD Hybrid Filtered', figure_hf)
                self.validation_writer.add_figure('GOF Hybrid Filtered', gof_hf)

