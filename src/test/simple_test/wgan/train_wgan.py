import torch
from app.trainer.simple.simple_trainer import SimpleTrainer
from common.common_nn import zerograd,zcat,modalite
from test.simple_test.wgan.strategy_discriminators_wgan import StrategyDiscriminatorWGAN
from tools.generate_noise import noise_generator
from common.common_model import gradient_penalty
from configuration import app

class WGAN(SimpleTrainer):
    def __init__(self,cv, trial=None):
        
        losses_disc = {
            'epochs':'',           'modality':'',
            'Dloss':'',     
            'Dloss_wgan_y':'',     'Dloss_wgan_zd':'',
            'Dloss_wgan_yz':''
        }
        losses_gens = {
            'epochs':'',           'modality':'',
            'Gloss':'',            'Gloss_wgan_y':'',
            'Gloss_wgan_zd':'',    'Gloss_wgan_yz':'',

            'Gloss_rec':'',        'Gloss_rec_y':'',     
            'Gloss_rec_zd':'',    
        }

        prob_disc = {
            'epochs':'',                'modality':'',
            'Dreal_y':'',               'Dfake_y':'',
            'Dreal_zd':'',              'Dfake_zd':'',

            'GPy':'',                   'GPzb':'',
            'GPyz':''
        }

        gradients_gens = {
            'epochs':'',    'modality':'',
            'F':'',         'Gy':'',
        }
        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dsyz':'',       'Dsy':'',
            'Dszb':'',
        }
        super(WGAN, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens,prob_disc   = prob_disc,
        strategy_discriminators = StrategyDiscriminatorWGAN, 
        gradients_gens = gradients_gens, gradients_disc = gradients_disc, actions=None, start_epoch=None)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,_ = batch
        for _ in range(1):
            zerograd([self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            zd_inp      = zyy
            y_inp       = zcat(y,wny) 
            
            y_gen       = self.gen_agent.Gy(zyy)
            zyy_F       = self.gen_agent.Fy(y_inp)
            zd_gen      = zyy_F

            Dreal_yz,Dfake_yz = self.disc_agent.discriminate_conjoint_yz(y,y_gen, zd_inp,zd_gen)
            GPyz= gradient_penalty(self.disc_agent.Dsyz, zcat(y,zd_gen), zcat(y_gen,zd_inp),app.DEVICE) \
                    if modality == 'train' else torch.zeros([])
            Dloss_wgan_yz = -(torch.mean(Dreal_yz.reshape(-1)) - torch.mean(Dfake_yz.reshape(-1)))+\
                            GPyz*app.LAMBDA_GP

            Dreal_y, Dfake_y = self.disc_agent.discriminate_marginal_y(y, y_gen)
            GPy = gradient_penalty(self.disc_agent.Dsy, y, y_gen,app.DEVICE) \
                    if modality == 'train' else torch.zeros([])
            Dloss_wgan_y= -(torch.mean(Dreal_y.reshape(-1)) - torch.mean(Dfake_y.reshape(-1))) +\
                                GPy*app.LAMBDA_GP
            
            Dreal_zd,Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen) 
            GPzb= gradient_penalty(self.disc_agent.Dszb, zd_inp, zd_gen, app.DEVICE) \
                    if modality == 'train' else torch.zeros([])
            Dloss_wgan_zd = -(torch.mean(Dreal_zd.reshape(-1)) - torch.mean(Dfake_zd.reshape(-1))) +\
                                 GPzb*app.LAMBDA_GP

            Dloss_wgan =  Dloss_wgan_yz + Dloss_wgan_y + Dloss_wgan_zd
            
            if modality == 'train':
                zerograd([self.disc_agent.optimizer])
                Dloss_wgan.backward(retain_graph=True)
                self.disc_agent.track_gradient(epoch)
                self.disc_agent.optimizer.step()
                
            self.losses_disc['epochs'       ] = epoch
            self.losses_disc['modality'     ] = modality

            self.losses_disc['Dloss'        ] = Dloss_wgan.tolist()
            self.losses_disc['Dloss_wgan_y' ] = Dloss_wgan_y.tolist()
            self.losses_disc['Dloss_wgan_zd'] = Dloss_wgan_zd.tolist()
            self.losses_disc_tracker.update()

            self.prob_disc['epochs'  ] = epoch
            self.prob_disc['modality'] = modality
            self.prob_disc['Dreal_y' ] = Dreal_y.mean().tolist()
            self.prob_disc['Dfake_y' ] = Dfake_y.mean().tolist()
            self.prob_disc['Dreal_zd'] = Dreal_zd.mean().tolist()
            self.prob_disc['Dfake_zd'] = Dfake_zd.mean().tolist()

            self.prob_disc['GPyz'    ] = GPyz.mean().tolist()
            self.prob_disc['GPy'     ] = GPy.mean().tolist()
            self.prob_disc['GPzb'    ] = GPzb.mean().tolist()
            self.prob_disc_tracker.update()

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,_ = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder, 
                self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            
            # 1. We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            zd_inp      = zyy
            y_inp       = zcat(y,wny)

            y_gen       = self.gen_agent.Gy(zyy)
            zd_gen      = self.gen_agent.Fy(y_inp)

            _, Dfake_yz = self.disc_agent.discriminate_conjoint_yz(y,y_gen, zd_inp,zd_gen)
            Gloss_wgan_yz = -(torch.mean(Dfake_yz.reshape(-1)))

            _, Dfake_y  = self.disc_agent.discriminate_marginal_y(y, y_gen)
            Gloss_wgan_y  = -(torch.mean(Dfake_y.reshape(-1)))

            _, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            Gloss_wgan_zd = -(torch.mean(Dfake_zd.reshape(-1)))

            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,app.NOISE)
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zd_gen)
            zd_rec      = self.gen_agent.Fy(y_gen)
            
            Gloss_rec_y = torch.mean(torch.abs(y-y_rec))
            Gloss_rec_zd= torch.mean(torch.abs(zd_inp-zd_rec))

            Gloss_rec   = Gloss_rec_y + Gloss_rec_zd

            Gloss       = Gloss_wgan_yz + Gloss_wgan_y + Gloss_wgan_zd + Gloss_rec*app.LAMBDA_IDENTITY

            if modality == 'train':
                zerograd([self.gen_agent.optimizer_encoder, self.gen_agent.optimizer_decoder, 
                    self.disc_agent.optimizer])
                Gloss.backward()
                self.gen_agent.track_gradient(epoch)
                self.gen_agent.optimizer_encoder.step()
                self.gen_agent.optimizer_decoder.step()
                
            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss'        ] = Gloss.tolist()
            self.losses_gens['Gloss_wgan_y' ] = Gloss_wgan_y.tolist()
            self.losses_gens['Gloss_wgan_zd'] = Gloss_wgan_zd.tolist()
            self.losses_gens['Gloss_rec'    ] = Gloss_rec.tolist()
            self.losses_gens['Gloss_rec_y'  ] = Gloss_rec_y.tolist()
            self.losses_gens['Gloss_rec_zd' ] = Gloss_rec_zd.tolist()
            self.losses_gen_tracker.update()

