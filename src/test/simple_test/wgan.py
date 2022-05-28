import torch
from app.trainer.simple.simple_trainer import SimpleTrainer
from common.common_nn import zerograd,zcat,modalite, clipweights
from tools.generate_noise import noise_generator
from configuration import app

class WGAN(SimpleTrainer):
    def __init__(self,cv, trial=None):
        losses_disc = {
            'epochs':'',           'modality':'',
            'Dloss':'',            'Dloss_wgan':'',        
            'Dloss_wgan_y':'',     'Dloss_wgan_zd':''
        }
        losses_gens = {
            'epochs':'',           'modality':'',
            'Gloss':'',            'Gloss_wgan':'',        'Gloss_wgan_y':'',
            'Gloss_wgan_zd':'',

            'Gloss_rec':'',        'Gloss_rec_y':'',     
            'Gloss_rec_zd':'',    
        }

        gradients_gens = {
            'epochs':'',    'modality':'',
            'Fxy':'',       'Gy':'',
        }
        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dy':'',        'Dsy':'',
            'Dzb':'',       'Dszb':'', 'Dyz':'',
            'Dszf':''
        }
        super(WGAN, self).__init__(cv, trial,losses_disc, losses_gens, gradients_gens, gradients_disc)
    
    def train_unic_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,_,zyy,zxy = batch
        for _ in range(5):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])
            # 1.1 We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            zd_inp      = zcat(zxy,zyy)
            y_inp       = zcat(y,wny) 

            y_gen       = self.gen_agent.Gy(zxy,zyy)
            zyy_F,zyx_F,*other = self.gen_agent.Fxy(y_inp)
            zd_gen      = zcat(zyx_F,zyy_F)

            Dreal_y, Dfake_y = self.disc_agent.discriminate_marginal_y(y, y_gen)
            Dloss_wgan_y     = -(torch.mean(Dreal_y) - torch.mean(Dfake_y))

            Dreal_zd, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            Dloss_wgan_zd    = -(torch.mean(Dreal_zd) - torch.mean(Dfake_zd))

            Dloss_wgan =  Dloss_wgan_y + Dloss_wgan_zd
            if modality == 'train':
                Dloss_wgan.backward()
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])

            clipweights(self.disc_agent.discriminators)
            self.losses_disc['epochs'       ] = epoch
            self.losses_disc['modality'     ] = modality
            self.losses_disc['Dloss'        ] = Dloss_wgan.tolist()
            self.losses_disc['Dloss_wgan'   ] = Dloss_wgan.tolist()
            self.losses_disc['Dloss_wgan_y' ] = Dloss_wgan_y.tolist()
            self.losses_disc['Dloss_wgan_zd'] = Dloss_wgan_zd.tolist()
            self.losses_disc_tracker.update()

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,_,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1.1 We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            zd_inp      = zcat(zxy,zyy)
            y_inp       = zcat(y,wny)

            y_gen       = self.gen_agent.Gy(zxy,zyy)
            zyy_F,zyx_F,*other = self.gen_agent.Fxy(y_inp)
            zd_gen      = zcat(zyx_F,zyy_F)

            _, Dfake_y = self.disc_agent.discriminate_marginal_y(y, y_gen)
            Gloss_y     = -(torch.mean(Dfake_y))

            _, Dfake_zd = self.disc_agent.discriminate_marginal_zd(zd_inp,zd_gen)
            Gloss_zd    = -(torch.mean(Dfake_zd))

            Gloss       = Gloss_y +Gloss_zd
            Gloss.backward()

