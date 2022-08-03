# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import streamlit as st
import common.ex_common_setup as cs
from test.simple_test.wgan.exp_wgan import StreamWGAN

@st.cache(allow_output_mutation=True)
def parse_args():
    cv = cs.setup()
    locals().update(cv)
    return cv

@st.cache(allow_output_mutation=True)
def load_model_and_dataset(cv, action, start_epoch):
    return StreamWGAN(cv=cv, trial=None, actions=action, start_epoch=start_epoch)

def main():
    u'''[SETUP] common variables/datasets''' 
    st.title("Pytorch Explorer")   
    cv = parse_args()
    stream_wgan = load_model_and_dataset(cv, action=True, start_epoch=1500)
    stream_wgan.stream_presentation()
  
main()
