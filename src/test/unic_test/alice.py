import torch
from app.trainer.unic.unic_trainer import UnicTrainer
from common.common_nn import zerograd,zcat,modalite, clipweights
from tools.generate_noise import noise_generator
from common.common_torch import *
from configuration import app

class ALICE(UnicTrainer):
    def __init__(self,cv, trial=None):
        losses_disc = {
            'epochs':'',           'modality':'',
            'Dloss':'',            'Dloss_ali':'',        'Dloss_ali_y':'',  
            'Dloss_ali_x':'',      'Dloss_marginal':'',   'Dloss_marginal_y':'',
            'Dloss_marginal_zd':'','Dloss_marginal_x':'', 'Dloss_marginal_zf':''
        }
        losses_gens = {
            'epochs':'',           'modality':'',
            'Gloss':'',            'Gloss_ali':'',        'Gloss_ali_x':'',
            'Gloss_ali_y':'',      'Gloss_marginal':'',   'Gloss_marginal_y':'',
            'Gloss_marginal_zd':'','Gloss_marginal_x':'', 'Gloss_marginal_zf':'',

            'Gloss_rec':'',        'Gloss_rec_y':'',      'Gloss_rec_x':'',
            'Gloss_rec_zd':'',     'Gloss_rec_zx':'',     'Gloss_rec_zxy':'',
            'Gloss_rec_x':'', 
        }

        prob_disc = {
            'epochs':'',            'modality':'',
            'Dreal_yz':'',          'Dfake_yz':'',
            'Dreal_xz':'',          'Dfake_xz':'',
            'Dreal_y':'',           'Dfake_y':'',
            'Dreal_x':'',           'Dfake_x':'',
            'Dreal_zd':'',          'Dfake_zd':'',
            'Dreal_zf':'',          'Dfake_zf':''
        }

        gradients_gens = {
            'epochs':'',    'modality':'',
            'Fxy':'',       'Gy':'',
        }
        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dy':'',        'Dx':'',   'Dsy':'',  'Dsx':'',
            'Dzb':'',       'Dszb':'', 'Dyz':'',  'Dzf':'',
            'Dszf':''
        }

        super(ALICE, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens, prob_disc  = prob_disc,
        gradients_gens = gradients_gens, gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,x,zyy,zxy = batch
        for _ in range(1):
            zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            modalite(self.gen_agent.generators,       mode = net_mode[0])
            modalite(self.disc_agent.discriminators,  mode = net_mode[1])

            # 1. We Generate conditional samples
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            zd_inp      = zcat(zxy,zyy)
            y_inp       = zcat(y,wny) 
            y_gen       = self.gen_agent.Gy(zxy,zyy)
            zyy_F,zyx_F,*other = self.gen_agent.Fxy(y_inp)
            zd_gen      = zcat(zyx_F,zyy_F)
            zxy_gen      = zyx_F
            
            Dreal_yz, Dfake_yz = self.disc_agent.discriminate_yz(y,y_gen,zd_inp,zd_gen)
            Dloss_ali_y =  self.bce_logit_loss(Dreal_yz,o1l(Dreal_yz)) +\
                            self.bce_logit_loss(Dfake_yz,o0l(Dfake_yz))

            x_gen       = self.gen_agent.Gy(zxy_gen,o0l(zyy))
            Dreal_xz, Dfake_xz = self.disc_agent.discriminate_xz(x,x_gen,zxy,zxy_gen)
            Dloss_ali_x = self.bce_logit_loss(Dreal_xz,o1l(Dreal_xz))+\
                            self.bce_logit_loss(Dfake_xz,o0l(Dfake_xz))
            
            Dloss_ali   = Dloss_ali_y + Dloss_ali_x
            
            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zyx_F, zyy_F)
            zyy_rec,zyx_rec = self.gen_agent.Fxy(y_gen)
            zd_rec      = zcat(zyx_rec,zyy_rec)

            wnx,*others = noise_generator(x.shape,zxy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            x_gen       = zcat(x_gen,wnx)
            x_rec       = self.gen_agent.Gy(zxy_gen,o0l(zyy))
            zxy_rec      = self.gen_agent.Fxy(x_gen)

            Dreal_y, Dfake_y        = self.disc_agent.discriminate_yy(y, y_rec)
            Dloss_cross_entropy_y   = self.bce_logit_loss(Dreal_y,o1l(Dreal_y))+\
                    self.bce_logit_loss(Dfake_y,o0l(Dfake_y))

            Dreal_zd, Dfake_zd      = self.disc_agent.discriminate_zzb(zd_inp,zd_rec)
            Dloss_cross_entropy_zd  = self.bce_logit_loss(Dreal_zd,o1l(Dreal_zd))+\
                    self.bce_logit_loss(Dfake_zd,o0l(Dfake_zd))

            Dreal_x, Dfake_x = self.disc_agent.discriminate_xx(x,x_rec)
            Dloss_cross_entropy_x   = self.bce_logit_loss(Dreal_x,o1l(Dreal_x))+\
                                        self.bce_logit_loss(Dfake_x,o0l(Dfake_x))

            Dfake_zf, Dreal_zf      = self.disc_agent.discriminate_zzf(zxy, zxy_gen)
            Dloss_cross_entropy_zf  = self.bce_logit_loss(Dreal_zf,o1l(Dreal_zf))+\
                                        self.bce_logit_loss(Dfake_zf,o0l(Dfake_zf))
            
            Dloss_cross_entropy     = Dloss_cross_entropy_y + Dloss_cross_entropy_zd+\
                                      Dloss_cross_entropy_x + Dloss_cross_entropy_zf
            
            Dloss = Dloss_ali + Dloss_cross_entropy

            if modality == 'train':
                Dloss.backward()
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])

            self.losses_disc['epochs'       ] = epoch
            self.losses_disc['modality'     ] = modality
            self.losses_disc['Dloss'        ] = Dloss.tolist()
            self.losses_disc['Dloss_ali_y'  ] = Dloss_ali_y.tolist()


    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(1):
            pass
