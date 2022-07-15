import torch
import streamlit as st
import matplotlib.pyplot as plt
from common.common_nn import zcat
from tools.generate_noise import noise_generator
from plot.plot_tools import plot_generate_classic
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
        st.header("Explore The WGAN with Gradient Penality")
        st.write("This page will give an over view of the WGAN GP training test.")
        
        self.stream_real_data_generation()
        self.stream_generate_values_from_z()
    
    def stream_generate_values_from_z(self):
        st.subheader("Generate signal From noise")
        result = st.button("Generate signal from Noise")
        if result:
            figure_gen, _ = plot_generate_classic(tag ='generated',
                Qec= self.gen_agent.Fy, Pdc= self.gen_agent.Gy,
                trn_set= self.data_tst_loader,
                pfx="vld_set_bb_unique",
                opt=self.opt, outf= self.opt.outf, save=False)

            for fig in figure_gen:
                st.write(fig)
            
    def stream_real_data_generation(self):
        st.subheader("Explore the generated data from the data")
        figure_bb, _ = plot_generate_classic(tag ='broadband',
            Qec= self.gen_agent.Fy, Pdc= self.gen_agent.Gy,
            trn_set= self.data_tst_loader,
            pfx="vld_set_bb_unique",
            opt=self.opt, outf= self.opt.outf, save=False)

        for fig in figure_bb:
            st.write(fig)
