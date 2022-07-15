# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import common.ex_common_setup as cs
from test.simple_test.wgan.exp_wgan import StreamWGAN
# from test.simple_test.alice import ALICE
def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)

    stream_wgan = StreamWGAN(cv)
    stream_wgan.stream_presentation()
  
main()
