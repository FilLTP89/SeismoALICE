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

        super(ALICE, self).__init__(cv, 
        trial       = None,
        losses_disc = losses_disc, 
        losses_gens = losses_gens, 
        gradients_gens = gradients_gens, 
        gradients_disc = gradients_disc)
    
    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(1):
            pass
    

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        y,zyy,zxy = batch
        for _ in range(1):
            pass
