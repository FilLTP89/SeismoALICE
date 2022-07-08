# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Train and Test AAE'''
u'''Required modules'''
# import warnings
# warnings.filterwarnings("ignore")
# from profile.profile_support import profile
import torch
import common.ex_common_setup as cs
from train.investigation.investigate_unic_tweaked import Investigator
from configuration import app




def run():
    cv = cs.setup()
    globals().update(cv)

    globals().update(opt.__dict__)

    app.logger.info("Investigation ")
    invest = Investigator(cv)

run()
    
