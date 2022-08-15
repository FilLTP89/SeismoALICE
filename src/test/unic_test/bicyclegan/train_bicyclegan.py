import torch
from app.trainer.unic.unic_trainer import UnicTrainer
from common.common_nn import zerograd,zcat,modalite
from plot.plot_tools import get_gofs, plot_generate_classic
from tools.generate_noise import noise_generator
from common.common_nn import get_accuracy
from common.common_torch import *
from configuration import app
from test.unic_test.bicyclegan.strategy_discriminator_bicyclegan import StrategyDiscriminatorBiCycleGAN
from test.unic_test.bicyclegan.strategy_generator_pix2pix import StrategyGeneratorBiCycleGAN

class BiCycleGAN(UnicTrainer):
    def __init__(self,cv, trial=None):
        losses_disc = {
            'epochs':'',                'modality':'',
            'Dloss':'', 
        }

        losses_gens = {
            'epochs':'',                'modality':'',
            'Gloss':''}

        prob_disc = {
            'epochs':'',                'modality':'',
        }

        gradients_gens = {
            'epochs':'',                'modality':'',
            'Fy':'',                    'Gxy':''
        }

        gradients_disc = {
            'epochs':'',                'modality':'',
            'Dxy':'',
        }

        super(BiCycleGAN, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens, prob_disc  = prob_disc,
        strategy_discriminator = StrategyDiscriminatorBiCycleGAN, 
        strategy_generator = StrategyGeneratorBiCycleGAN, 
        gradients_gens = gradients_gens, 
        gradients_disc = gradients_disc)


    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                    self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1. cVAE-GAN
            mu, logvar  = self.gen_agent.Fy(y)
            zxy_gen     = self.gen_agent.reparametrization_trick(mu, logvar)
            y_gen       = self.gen_agent.Gxy(x,zxy_gen)

            Dreal_VAE_GAN, Dfake_VAE_GAN = self.disc_agent.discriminate_marginal_y(y, y_gen)
            Dloss_VAE_GAN = self.bce_logit_loss(Dreal_VAE_GAN.reshape(-1), o1l(Dreal_VAE_GAN.reshape(-1)))+\
                self.bce_logit_loss(Dfake_VAE_GAN.reshape(-1),o0l(Dfake_VAE_GAN.reshape(-1)))

            if modality == 'train':
                zerograd([self.disc_agent.optimizer_vae])
                Dloss_VAE_GAN.backward()
                self.disc_agent.optimizer_vae.step()

            # 2. cLR-GAN
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                    self.disc_agent.optimizer])
            _y_gen = self.gen_agent.Gxy(y,zxy)
            Dreal_LR_GAN, Dfake_LR_GAN = self.disc_agent.discriminate_marginal_zd(y,_y_gen)
            Dloss_LR_GAN = self.bce_logit_loss(Dreal_LR_GAN.reshape(-1), o1l(Dreal_LR_GAN.reshape(-1))) +\
                    self.bce_logit_loss(Dfake_LR_GAN.reshape(-1), o0l(Dfake_LR_GAN.reshape(-1)))
                
            if modality == 'train':
                zerograd([self.disc_agent.optimizer_lr])
                Dloss_LR_GAN.backward()
                self.disc_agent.optimizer_lr.step()
                self.disc_agent.track_gradient(epoch)
            
            self.losses_disc['epochs'       ] = epoch
            self.losses_disc['modality'     ] = modality
            self.losses_disc['Dloss_VAE_GAN'] = Dloss_VAE_GAN.tolist()
            self.losses_disc['Dloss_LR_GAN' ] = Dloss_LR_GAN.tolist()
            self.losses_disc_tracker.update()

            self.prob_disc['Dreal_VAE_GAN'  ] = torch.sigmoid(Dreal_VAE_GAN).mean().tolist()
            self.prob_disc['Dfake_VAE_GAN'  ] = torch.sigmoid(Dfake_VAE_GAN).mean().tolist()
            self.prob_disc['Dreal_LR_GAN'   ] = torch.sigmoid(Dreal_LR_GAN).mean().tolist()
            self.prob_disc['Dfake_LR_GAN'   ] = torch.sigmoid(Dfake_LR_GAN).mean().tolist()
            self.prob_disc_tracker.update()

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder, 
                            self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1. cVAE-GAN
            mu, logvar  = self.gen_agent.Fy(y)
            zxy_gen     = self.gen_agent.reparametrization_trick(mu, logvar)
            y_gen       = self.gen_agent.Gxy(x,zxy_gen)

            Gloss_pixel = self.l1_loss(y_gen,y)
            Gloss_KL    = 0.5 * torch.sum(torch.exp(logvar) + mu**2 -logvar -1)
            _,Dfake_VAE_GAN = self.disc_agent.discriminate_marginal_y(y, y_gen)
            Gloss_VAE_GAN   = self.bce_logit_loss(Dfake_VAE_GAN.reshape(-1), o1l(Dfake_VAE_GAN.reshape(-1)))

            # 2. cLR-GAN
            _y_gen = self.gen_agent.Gxy(x,zxy)
            _, Dfake_LR_GAN = self.disc_agent.discriminate_marginal_zd(y,_y_gen)
            Gloss_LR_GAN = self.bce_logit_loss(Dfake_LR_GAN.reshape(-1), o1l(Dfake_LR_GAN.reshape(-1)))

            Gloss_GE = Gloss_VAE_GAN + Gloss_LR_GAN + Gloss_pixel*0.5 + Gloss_KL*10.
            if modality == 'train':
                zerograd([self.gen_agent.optimizer_encoder])
                Gloss_GE.backward(retain_graph=True)
                self.gen_agent.optimizer_decoder.step()

            # 3. Latent L1 loss
            _mu, _ = self.gen_agent.Fy(_y_gen)
            Gloss_Latent = self.l1_loss(_mu, zxy)*10.

            if modality == 'train':
                zerograd([self.gen_agent.optimizer_decoder])
                Gloss_Latent.backward()
                self.gen_agent.optimizer_encoder.step()
                self.gen_agent.track_gradient(epoch)
            
            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss_GE'     ] = Gloss_GE.tolist()
            self.losses_gens['Gloss_pix2pix'] = Gloss_VAE_GAN.tolist()
            self.losses_gens['Gloss_LR_GAN '] = Gloss_LR_GAN.tolist()
            self.losses_gens['Gloss_pixel'  ] = Gloss_pixel.tolist()
            self.losses_gens['Gloss_KL'     ] = Gloss_KL.tolist()
            self.losses_gen_tracker.update()


