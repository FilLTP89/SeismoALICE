import streamlit as st
from app.trainer.simple.simple_trainer import SimpleTrainer
from configuration import app


class StreamWGAN(SimpleTrainer):
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
            'F':'',         'Gy':'',
        }
        gradients_disc = {
            'epochs':'',    'modality':'',
            'Dsy':'',       'Dszb':''
        }
        super(StreamWGAN, self).__init__(cv, trial = None,
        losses_disc = losses_disc, losses_gens = losses_gens,prob_disc   = prob_disc,
        gradients_gens = gradients_gens, gradients_disc = gradients_disc, actions=True, start_epoch=5000)

    def train_discriminators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        pass

    def train_generators(self,batch,epoch,modality,net_mode,*args,**kwargs):
        pass

    def stream_presentation(self):
        st.write("""
            ### Explore WGAN dataset and Generations' values

            This page will give an over view of the WGAN GP training test.

            """)