# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import streamlit as st
import common.ex_common_setup as cs
breakpoint()
from test.simple_test.wgan.exp_wgan import StreamWGAN
# from test.simple_test.alice import ALICE
def main():
    u'''[SETUP] common variables/datasets'''
    breakpoint()
    @st.cache
    def parse_args():
        cv = cs.setup()
        locals().update(cv)
        return cv

    cv = parse_args()

    stream_wgan = StreamWGAN(cv)
    stream_wgan.stream_presentation()
  
main()
