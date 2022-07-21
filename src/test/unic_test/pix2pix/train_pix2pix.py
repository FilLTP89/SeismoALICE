import torch
from app.trainer.unic.unic_trainer import UnicTrainer
from common.common_nn import zerograd,zcat,modalite
from tools.generate_noise import noise_generator
from common.common_torch import *
from configuration import app
from test.simple_test.pix2pix.strategy_discriminator_pix2pix import StrategyDiscriminatoPix2Pix
from test.simple_test.pix2pix.strategy_generator_pix2pix import StrategyGeneratorPix2Pix

class Pix2Pix(UnicTrainer):
    def __init__(self,cv, trial=None):
        losses_disc = {
            'epochs':'',                'modality':'',
            'Dloss':'',
            'Dloss_xy':'',              'Dloss_zd':'', 
        }

        losses_gens = {
            'epochs':'',                'modality':'',
            'Gloss':'',                 'Gloss_xy':'',
            'Gloss_zd':'',      

            'Gloss_rec':'',             'Gloss_rec_y':'',
            'Gloss_rec_zd':'',
        }

        prob_disc = {
            'epochs':'',            'modality':'',
            'Dreal_xy':'',          'Dfake_xy':'',
            'Dreal_x':'',           'Dfake_x':'',
            'Dreal_zd':'',          'Dfake_zd':'',
        }

        gradients_gens = {
            'epochs':'',    'modality':'',
            'Fx':'',        'Gy':'',
        }

        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dsxy':'',      'Dszb':'',
        }

        super(Pix2Pix, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens, prob_disc  = prob_disc,
        strategy_discriminator = StrategyDiscriminatoPix2Pix, 
        strategy_generator = StrategyGeneratorPix2Pix, 
        gradients_gens = gradients_gens, 
        gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            wnx,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            x_inp       =  zcat(x,wnx)
            zd_inp      = zcat(zyy)

            # 1. Pix2Pix
            zd_gen      = self.gen_agent.Fxy(x_inp)
            y_gen       = self.gen_agent.Gy(zd_gen)
            
            Dreal_xy, Dfake_xy = self.disc_agent.discriminate_conjoint_yz(x,y,y_gen)
            Dloss_xy    = self.bce_logit_loss(Dreal_xy.reshape(-1),o1l(Dfake_xy.reshape(-1))) +\
                            self.bce_logit_loss(Dfake_xy.reshape(-1),o0l(Dfake_xy.reshape(-1)))
            
            Dreal_zd, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zxy,zd_gen)
            Dloss_zd    = self.bce_logit_loss(Dreal_zd.reshape(-1),o1l(Dfake_zd.reshape(-1)))+\
                            self.bce_logit_loss(Dfake_zd.reshape(-1),o0l(Dfake_zd.reshape(-1)))
            
            # 2. Summation of losses
            Dloss_pix2pix      = Dloss_xy + Dloss_zd
            Dloss              = Dloss_pix2pix
            
            if modality == 'train':
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
                Dloss.backward()
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
            
            self.losses_disc['epochs'   ] = epoch
            self.losses_disc['modality' ] = modality
            self.losses_disc['Dloss'    ] = Dloss.tolist()
            self.losses_disc['Dloss_xy' ] = Dloss_xy.tolist()
            self.losses_disc['Dloss_zd' ] = Dloss_zd.tolist()
            self.losses_disc['Dloss_pix2pix'] = Dloss_pix2pix.tolist()
            self.losses_disc_tracker.update()

            self.prob_disc['epochs'  ] = epoch
            self.prob_disc['modality'] = modality
            self.prob_disc['Dreal_xy'] = Dreal_xy.mean().tolist()
            self.prob_disc['Dfake_xy'] = Dfake_xy.mean().tolist()
            self.prob_disc['Dreal_zd'] = Dreal_zd.mean().tolist()
            self.prob_disc['Dfake_zd'] = Dfake_zd.mean().tolist()
            self.prob_disc_tracker.update()
            

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                         self.disc_agent.optimizer])
            
            # 1. Pix2Pix
            wnx,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            x_inp       =  zcat(x,wnx)
            zd_inp      = zcat(zyy)

            zd_gen      = self.gen_agent.Fx(x_inp)
            y_gen       = self.gen_agent.Gy(zd_gen)
            
            _, Dfake_xy = self.disc_agent.discriminate_xy(x,y,y_gen)
            Gloss_xy    = self.bce_logit_loss(Dfake_xy.reshape(-1),o1l(Dfake_xy.reshape(-1)))
            
            _, Dfake_zd  = self.disc_agent.discriminate_marginal_zd(zxy,zd_gen)
            Gloss_zd    = self.bce_logit_loss(Dfake_zd.reshape(-1),o1l(Dfake_zd.reshape(-1)))

            # 2. Reconstruction
            Gloss_rec_y = torch.mean(torch.abs(y-y_gen))
            Gloss_rec_zd= torch.mean(torch.abs(zxy -zd_gen))
            
            # 3. Summation of losses
            Gloss_pix2pix   = Gloss_xy + Gloss_zd
            Gloss_rec       = Gloss_rec_y + Gloss_rec_zd
            Gloss           = Gloss_pix2pix + Gloss_rec*app.LAMBDA_IDENTITY

            if modality == 'train':
                zerograd([self.gen_agent.optimize_encoder, self.gen_agent.optimizer_decoder,
                         self.disc_agent.optimizer])
                Gloss.backward()
                self.gen_agent.optimizer.step()
                self.gen_agent.track_gradient(epoch)

            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss'        ] = Gloss.tolist()
            self.losses_gens['Gloss_pix2pix'] = Gloss_pix2pix.tolist()
            self.losses_gens['Gloss_rec'    ] = Gloss_rec.tolist()
            self.losses_gens['Gloss_xy'     ] = Gloss_xy.tolist()
            self.losses_gens['Gloss_rec_y'  ] = Gloss_rec_y.tolist()
            self.losses_gens['Gloss_rec_zd' ] = Gloss_rec_zd.tolist()
            self.losses_gen_tracker.update()


            



