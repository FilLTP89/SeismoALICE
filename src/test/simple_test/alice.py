import torch
from app.trainer.simple.simple_trainer import SimpleTrainer
from common.common_nn import zerograd,zcat,modalite, Exp
from tools.generate_noise import noise_generator
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
            'epochs':'',    'modality':'',
            'Fxy':'',       'Gy':'',
        }
        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dy':'',        'Dyy':'',
            'Dzb':'',       'Dzzb':'', 'Dyz':'',
        }
        super(ALICE, self).__init__(cv, trial = None, 
            losses_disc = losses_disc, losses_gens = losses_gens, prob_disc  = prob_disc,
            gradients_gens = gradients_gens, gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
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
            
            Dreal_yz, Dfake_yz = self.disc_agent.discriminate_yz(y,y_gen,zd_inp,zd_gen)
            Dloss_ali_y =  self.bce_logit_loss(Dreal_yz,o1l(Dreal_yz)) +\
                             self.bce_logit_loss(Dfake_yz,o0l(Dfake_yz))

            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zyx_F, zyy_F)

            zyy_rec,zyx_rec = self.gen_agent.Fy(y_gen)
            zd_rec      = zcat(zyx_rec,zyy_rec)

            Dreal_y, Dfake_y        = self.disc_agent.discriminate_yy(y, y_rec)
            Dloss_cross_entropy_y   = self.bce_logit_loss(Dreal_y,o1l(Dreal_y))+\
                    self.bce_logit_loss(Dfake_y,o0l(Dfake_y))

            Dreal_zd, Dfake_zd      = self.disc_agent.discriminate_zzb(zd_inp,zd_rec)
            Dloss_cross_entropy_zd  = self.bce_logit_loss(Dreal_zd,o1l(Dreal_zd))+\
                    self.bce_logit_loss(Dfake_zd,o0l(Dfake_zd))

            Dloss_cross_entropy = Dloss_cross_entropy_y + Dloss_cross_entropy_zd
            Dloss               = Dloss_ali_y + Dloss_cross_entropy
            
            if modality == 'train':
                Dloss.backward()
                self.disc_agent.optimizer.step()
                self.disc_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])

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
            self.prob_disc['Dreal_yz'] = Dreal_yz.mean().tolist()
            self.prob_disc['Dfake_yz'] = Dfake_yz.mean().tolist()
            self.prob_disc['Dreal_y' ] = Dreal_y.mean().tolist()
            self.prob_disc['Dfake_y' ] = Dfake_y.mean().tolist()
            self.prob_disc['Dreal_zd'] = Dreal_zd.mean().tolist()
            self.prob_disc['Dfake_zd'] = Dfake_zd.mean().tolist()
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

            Dreal_yz, Dfake_yz  = self.disc_agent.discriminate_yz(y, y_gen,zd_inp,zd_gen)
            Gloss_ali_y = self.bce_logit_loss(Dreal_yz,o1l(Dreal_yz)) +\
                         self.bce_logit_loss(Dfake_yz,o0l(Dfake_yz))

            # 2. Reconstruction of signal distributions
            wny,*others = noise_generator(y.shape,zyy.shape,app.DEVICE,{'mean':0., 'std':self.std})
            y_gen       = zcat(y_gen,wny)
            y_rec       = self.gen_agent.Gy(zyx_F, zyy_F)

            zyy_rec,zyx_rec = self.gen_agent.Fy(y_gen)
            zd_rec      = zcat(zyx_rec,zyy_rec)

            Gloss_rec_y = torch.mean(torch.abs(y-y_rec))
            Gloss_rec_zd= torch.mean(torch.abs(zd_inp-zd_rec))
            
            _, Dfake_y        = self.disc_agent.discriminate_yy(y, y_rec)
            Gloss_cross_entropy_y   = self.bce_logit_loss(Dfake_y,o1l(Dfake_y))

            _, Dfake_zd      = self.disc_agent.discriminate_zzb(zd_inp,zd_rec)
            Gloss_cross_entropy_zd  = self.bce_logit_loss(Dfake_zd,o1l(Dfake_zd))

            Gloss_rec           = Gloss_rec_y + Gloss_rec_zd
            Gloss_cross_entropy = Gloss_cross_entropy_y + Gloss_cross_entropy_zd
            Gloss               = Gloss_ali_y + Gloss_cross_entropy +Gloss_rec
            
            if modality == 'train':
                Gloss.backward()
                self.gen_agent.optimizer.step()
                self.gen_agent.track_gradient(epoch)
                zerograd([self.gen_agent.optimizer, self.disc_agent.optimizer])
            
            self.losses_gens['epochs'       ] = epoch
            self.losses_gens['modality'     ] = modality
            self.losses_gens['Gloss'        ] = Gloss.tolist()
            self.losses_gens['Gloss_ali_y'  ] = Gloss_ali_y.tolist()
            self.losses_gens['Gloss_rec'    ] = 0 #Gloss_rec.tolist()
            self.losses_gens['Gloss_rec_y'  ] = 0 #Gloss_rec_y.tolist()
            self.losses_gens['Gloss_rec_zd' ] = 0 #Gloss_rec_zd.tolist()
            self.losses_gens['Gloss_cross_entropy'] = Gloss_cross_entropy.tolist()
            self.losses_gens['Gloss_cross_entropy_y' ] = Gloss_cross_entropy_y.tolist()
            self.losses_gens['Gloss_cross_entropy_zd'] = Gloss_cross_entropy_zd.tolist()
            self.losses_gen_tracker.update()

