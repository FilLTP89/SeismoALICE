import torch
import streamlit as st
import matplotlib.pyplot as plt
from common.common_nn import zcat
from tools.generate_noise import noise_generator
from app.trainer.simple.simple_trainer import SimpleTrainer
from configuration import app

class StreamWGAN(SimpleTrainer):
    def __init__(self,cv, trial=None):
        breakpoint()
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
            ### Explore The WGAN with Gradient Penality

            This page will give an over view of the WGAN GP training test.

            """)
        st.write(""" Generate noise """)
        result = st.button("Generate signal from Noise")

        if result:
            z_tar       = torch.randn(1,1,512).to(app.DEVICE)
            st.write("shape of noises :")
            st.write(z_tar.shape)
            
            y_gen       = self.gen_agent.Gy(z_tar)
            st.write("shape of y")
            st.write(y_gen.shape)
            wny,*others = noise_generator(y_gen.shape,z_tar.shape,app.DEVICE,app.NOISE)
            z_gen       = self.gen_agent.Fy(zcat(y_gen,wny))
            
            st.write(" Comparaison between noise")
            z_gen = z_gen.cpu().data.numpy()
            z_tar = z_tar.cpu().data.numpy()
            
            plt.figure(figsize=(6,6))
            fig1, ax = plt.subplots()
            ax.hist(z_gen[0,0,:], bins=30, density=True, label='cal', fc=(0.8, 0, 0, 1))
            ax.hist(z_tar[0,0,:], bins=10, density=True, label='targ',fc=(1., 0.8, 0, 0.8))
            ax.set_xlim([-4,4])
            ax.set_ylim([0,0.5])
            ax.set_xlabel('z')

            st.pyplot(fig1)

            st.write(f" Generate images from noise shape {y_gen.shape}")
            plt.figure(figsize=(12,8))
            _, c, w = y_gen.shape
            y_gen = y_gen.cpu().data.numpy()

            fig2, ax = plt.subplots(1,c)
            for i in range(c):
                ax[i].plot(y_gen[0,i,:])
                ax[i].set_xlabel('t')
                ax[i].set_ylabel('A(t)')
                ax[i].set_ylim([-1,1])
                ax[i].set_xlim([0,w])

            st.pyplot(fig2)
        # self.stream_generate_values_from_z()
    
    def stream_generate_values_from_z(self):
        pass
            

