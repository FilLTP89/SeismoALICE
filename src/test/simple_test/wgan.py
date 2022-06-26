import torch
from app.trainer.simple.simple_trainer import SimpleTrainer
from common.common_nn import zerograd,zcat,modalite, clipweights
from tools.generate_noise import noise_generator
from common.common_model import gradient_penalty
from configuration import app

class WGAN(SimpleTrainer):
    def __init__(self,cv, trial=None):
        
        losses_disc = {
            'epochs':'',           'modality':'',
            'Dloss':'',        
            'Dloss_wgan_y':'',     'Dloss_wgan_zd':''
        }
        losses_gens = {
            'epochs':'',           'modality':'',
            'Gloss':'',            'Gloss_wgan_y':'',
            'Gloss_wgan_zd':'',

            'Gloss_rec':'',        'Gloss_rec_y':'',     
            'Gloss_rec_zd':'',    
        }

        prob_disc = {
            'epochs':'',                'modality':'',
            'Dreal_y':'',               'Dfake_y':'',
            'Dreal_zd':'',              'Dfake_zd':'',
            'GPy':'',                   'GPzb':''
        }

        gradients_gens = {
            'epochs':'',    'modality':'',
            'Fxy':'',       'Gy':'',
        }
        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dsy':'',       'Dszb':''
        }
        super(WGAN, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens,prob_disc   = prob_disc,
        gradients_gens = gradients_gens, gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(5):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            breakpoint()
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            zd_inp      = zcat(zxy,zyy)
            y_inp       = zcat(y,wny) 
            
            y_gen       = self.gen_agent.Gy(zxy,zyy)
            zyy_F,zyx_F,*other = self.gen_agent.Fy(y_inp)
            zd_gen      = zcat(zyx_F,zyy_F)

            Dreal_y, Dfake_y = self.disc_agent.discriminate_marginal_y(y, y_gen)
            GPy              = gradient_penalty(self.disc_agent.Dsy, y, y_gen,app.DEVICE)
            Dloss_wgan_y     = -(torch.mean(Dreal_y.reshape(-1)) - torch.mean(Dfake_y.reshape(-1))) +\
                                app.LAMBDA_GP*GPy
            
            Dreal_zd, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            GPzb             = gradient_penalty(self.disc_agent.Dszb, zd_inp, zd_gen, app.DEVICE)
            Dloss_wgan_zd    = -(torch.mean(Dreal_zd.reshape(-1)) - torch.mean(Dfake_zd.reshape(-1)))+\
                                app.LAMBDA_GP*GPzb

            Dloss_wgan =  Dloss_wgan_y + Dloss_wgan_zd
            
            if modality == 'train':
                # Dfake_y.register_hook(lambda grad: print(grad))
                # self.disc_agent.Dszb.module.cnn[8].weight.register_hook(lambda grad: print(grad))
                Dloss_wgan.backward(retain_graph=True)
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])

            # no clipweights spectral_norm is implemented
            
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
            self.prob_disc['GPy'     ] = GPy.mean().tolist()
            self.prob_disc['GPzb'    ] = GPzb.mean().tolist()
            self.prob_disc_tracker.update()

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1. We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            zd_inp      = zcat(zxy,zyy)
            y_inp       = zcat(y,wny)

            y_gen       = self.gen_agent.Gy(zxy,zyy)
            zyy_F,zyx_F,*other = self.gen_agent.Fy(y_inp)
            zd_gen      = zcat(zyx_F,zyy_F)

            _, Dfake_y  = self.disc_agent.discriminate_marginal_y(y, y_gen)
            Gloss_wgan_y= -(torch.mean(Dfake_y.reshape(-1)))

            _, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            Gloss_wgan_zd= -(torch.mean(Dfake_zd.reshape(-1)))

            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zyx_F, zyy_F)

            zyy_rec,zyx_rec = self.gen_agent.Fy(y_gen)
            zd_rec      = zcat(zyx_rec,zyy_rec)

            Gloss_rec_y = torch.mean(torch.abs(y-y_rec))
            Gloss_rec_zd= torch.mean(torch.abs(zd_inp-zd_rec))

            Gloss_rec   = Gloss_rec_y + Gloss_rec_zd

            Gloss = Gloss_wgan_y + Gloss_wgan_zd + Gloss_rec
            if modality == 'train':
                Gloss.backward()
                self.gen_agent.optimizer.step()
                self.gen_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            
            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss'        ] = Gloss.tolist()
            self.losses_gens['Gloss_wgan_y' ] = Gloss_wgan_y.tolist()
            self.losses_gens['Gloss_wgan_zd'] = Gloss_wgan_zd.tolist()
            self.losses_gens['Gloss_rec'    ] = Gloss_rec.tolist()
            self.losses_gens['Gloss_rec_y'  ] = Gloss_rec_y.tolist()
            self.losses_gens['Gloss_rec_zd' ] = Gloss_rec_zd.tolist()
            self.losses_gen_tracker.update()

