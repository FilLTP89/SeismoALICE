# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import streamlit as st
import common.ex_common_setup as cs
from test.simple_test.wgan.exp_wgan import StreamWGAN

def main():
    u'''[SETUP] common variables/datasets'''

    @st.cache(allow_output_mutation=True)
    def parse_args():
        cv = cs.setup()
        locals().update(cv)
        return cv

    cv = parse_args()
    stream_wgan = StreamWGAN(cv)
    stream_wgan.stream_presentation()
  
main()
