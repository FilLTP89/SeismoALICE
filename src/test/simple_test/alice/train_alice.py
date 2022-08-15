import torch
from app.trainer.simple.simple_trainer import SimpleTrainer
from common.common_nn import zerograd,zcat,modalite
from tools.generate_noise import noise_generator
from test.simple_test.alice.strategy_discriminator_alice import StrategyDiscriminatorALICE
from test.simple_test.alice.strategy_generator_alice import StrategyGeneratorALICE
from common.common_torch import *
from configuration import app

class ALICE(SimpleTrainer):
    def __init__(self,cv, trial=None):
        losses_disc = {
            'epochs':'',           'modality':'',
            'Dloss':'',            'Dloss_cross_entropy':'',             
            'Dloss_ali_y':'',      'Dloss_cross_entropy_y':''
            'Dloss_cross_entropy_zd'
        }

        losses_gens = {
            'epochs':'',                'modality':'',
            'Gloss':'',                 'Gloss_cross_entropy':'',
            'Gloss_ali_y':'',
            'Gloss_cross_entropy_y':'', 'Gloss_cross_entropy_zd':'',
            'Gloss_rec':'',             'Gloss_rec_y':'',     
            'Gloss_rec_zd':'',    
        }

        prob_disc = {
            'epochs':'',                'modality':'',
            'Dreal_yz':'',              'Dfake_yz':'',
            'Dreal_y':'',               'Dfake_y':'',
            'Dreal_zd':'',              'Dfake_zd':'',
        }

        gradients_gens = {
            'epochs':'',                'modality':'',
            'Fy':'',                    'Gy':'',
        }

        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dsy':'',        'Dyy':'',
            'Dszb':'',       'Dzzb':'', 
            'Dyz':'',
        }
        super(ALICE, self).__init__(cv, trial = None, 
            losses_disc = losses_disc, losses_gens = losses_gens, prob_disc  = prob_disc,
            strategy_discriminator = StrategyDiscriminatorALICE,
            strategy_generator = StrategyGeneratorALICE,
            gradients_gens = gradients_gens, gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                         self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            
            # 1. We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            zd_inp      = zxy
            y_inp       = zcat(y,wny) 
            y_gen       = self.gen_agent.Gy(zd_inp)
            zd_gen      = self.gen_agent.Fy(y_inp)
            Dreal_yz, Dfake_yz      = self.disc_agent.discriminate_conjoint_yz(y,y_gen,zd_inp,zd_gen)
            Dloss_ali_y             = 0.5*self.bce_logit_loss(Dreal_yz.reshape(-1),o1l(Dreal_yz.reshape(-1)))+\
                        0.5*self.bce_logit_loss(Dfake_yz.reshape(-1),o0l(Dfake_yz.reshape(-1)))

            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zd_gen)
            zd_rec      = self.gen_agent.Fy(y_gen)
            
            Dreal_y, Dfake_y        = self.disc_agent.discriminate_cross_entropy_y(y, y_rec)
            Dloss_cross_entropy_y   = 0.5*self.bce_logit_loss(Dreal_y.reshape(-1),o1l(Dreal_y.reshape(-1)))+\
                        0.5*self.bce_logit_loss(Dfake_y.reshape(-1),o0l(Dfake_y.reshape(-1)))
            
            Dreal_zd, Dfake_zd      = self.disc_agent.discriminate_cross_entropy_zd(zd_inp,zd_rec)
            Dloss_cross_entropy_zd  = 0.5*self.bce_logit_loss(Dreal_zd.reshape(-1),o1l(Dreal_zd.reshape(-1)))+\
                        0.5*self.bce_logit_loss(Dfake_zd.reshape(-1),o0l(Dfake_zd.reshape(-1)))
            Dloss_cross_entropy = Dloss_cross_entropy_y + Dloss_cross_entropy_zd
            Dloss               = Dloss_ali_y + Dloss_cross_entropy
            
            if modality == 'train':
                zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                            self.disc_agent.optimizer])
                Dloss.backward(retain_graph=True)
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
                
            self.losses_disc['epochs'       ] = epoch
            self.losses_disc['modality'     ] = modality
            self.losses_disc['Dloss'        ] = Dloss.tolist()
            self.losses_disc['Dloss_ali_y'  ] = Dloss_ali_y.tolist()
            self.losses_disc['Dloss_cross_entropy'   ] = Dloss_cross_entropy.tolist()
            self.losses_disc['Dloss_cross_entropy_y' ] = Dloss_cross_entropy_y.tolist()
            self.losses_disc['Dloss_cross_entropy_zd'] = Dloss_cross_entropy_zd.tolist()            
            self.losses_disc_tracker.update()

            self.prob_disc['epochs'  ] = epoch
            self.prob_disc['modality'] = modality
            self.prob_disc['Dreal_yz'] = torch.sigmoid(Dreal_yz).mean().tolist()
            self.prob_disc['Dfake_yz'] = torch.sigmoid(Dfake_yz).mean().tolist()
            self.prob_disc['Dreal_y' ] = torch.sigmoid(Dreal_y).mean().tolist()
            self.prob_disc['Dfake_y' ] = torch.sigmoid(Dfake_y).mean().tolist()
            self.prob_disc['Dreal_zd'] = torch.sigmoid(Dreal_zd).mean().tolist()
            self.prob_disc['Dfake_zd'] = torch.sigmoid(Dfake_zd).mean().tolist()
            self.prob_disc_tracker.update()
            
    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                    self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            
            # 1. We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            zd_inp      = zxy
            y_inp       = zcat(y,wny)

            y_gen       = self.gen_agent.Gy(zd_inp)
            zd_gen      = self.gen_agent.Fy(y_inp)

            Dreal_yz, Dfake_yz  = self.disc_agent.discriminate_conjoint_yz(y, y_gen,zd_inp,zd_gen)
            Gloss_ali_y         = 0.005*self.bce_logit_loss(Dreal_yz.reshape(-1),o0l(Dreal_yz.reshape(-1))) +\
                        0.005*self.bce_logit_loss(Dfake_yz.reshape(-1),o1l(Dfake_yz.reshape(-1)))

            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zd_gen)
            zd_rec      = self.gen_agent.Fy(y_gen)

            Gloss_rec_y = self.l1_loss(y,y_rec)
            Gloss_rec_zd= self.l1_loss(zd_inp,zd_rec)
            
            _, Dfake_y              = self.disc_agent.discriminate_cross_entropy_y(y, y_rec)
            Gloss_cross_entropy_y   = 0.005*self.bce_logit_loss(Dfake_y.reshape(-1),o1l(Dfake_y.reshape(-1)))

            _, Dfake_zd             = self.disc_agent.discriminate_cross_entropy_zd(zd_inp,zd_rec)
            Gloss_cross_entropy_zd  = 0.005*self.bce_logit_loss(Dfake_zd.reshape(-1),o1l(Dfake_zd.reshape(-1)))

            Gloss_rec           = Gloss_rec_y + Gloss_rec_zd
            Gloss_cross_entropy = Gloss_cross_entropy_y + Gloss_cross_entropy_zd
            Gloss               = Gloss_ali_y + Gloss_cross_entropy + Gloss_rec
           
            if modality == 'train':
                zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder,
                            self.disc_agent.optimizer])
                Gloss.backward()
                self.gen_agent.optimizer_encoder.step()
                self.gen_agent.optimizer_decoder.step()
                self.gen_agent.track_gradient(epoch)
                
            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss'        ] = Gloss.tolist()
            self.losses_gens['Gloss_ali_y'  ] = Gloss_ali_y.tolist()
            self.losses_gens['Gloss_rec'    ] = Gloss_rec.tolist()
            self.losses_gens['Gloss_rec_y'  ] = Gloss_rec_y.tolist()
            self.losses_gens['Gloss_rec_zd' ] = Gloss_rec_zd.tolist()
            self.losses_gens['Gloss_cross_entropy'] = Gloss_cross_entropy.tolist()
            self.losses_gens['Gloss_cross_entropy_y' ] = Gloss_cross_entropy_y.tolist()
            self.losses_gens['Gloss_cross_entropy_zd'] = Gloss_cross_entropy_zd.tolist()
            self.losses_gen_tracker.update()

