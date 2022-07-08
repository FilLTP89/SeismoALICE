# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Train and Test AAE'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
from profile.profile_support import profile
import common.ex_common_setup as cs
import train.test_trainer_bb as aat
import torch.multiprocessing as mp
import torch.distributed as dist
u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2019, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

#file: aae_drive.py
@profile
def run():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()

    # And therefore this latter is become accessible to the methods in this class
    globals().update(cv)
    # define as global opt and passing it as a dictonnary here
    globals().update(opt.__dict__)

    mp.set_start_method('spawn')
    args = [cv, aat]
    # mp.spawn(session, nprocs = opt.ngpu, args=(args,))
    session(opt)
    # session(cv,aat,1)
    

def session(opt):
    # globals().update(cv)
    DCA = aat.trainer(opt)
    u'''[TRAIN] neural networks'''
    DCA.train()
    # u'''[GENERATE] samples'''
    # DCA.generate()
    # u'''[TEST] discrimination'''
    # DCA.discriminate()
    # u'''[STAT] spanning prob distribution'''
    # DCA.compare()
# def dummy(gpu, args):
#     print(f'this the execution fonction for {gpu}')
#     opt = args
#     rank = opt.nr * opt.ngpu + gpu
#     print(
#         f"Rank {rank + 1}/{opt.world_size} process initialized.\n"
#     )
#     dist.init_process_group(
#             backend       ='nccl',
#             init_method   ='env://',
#             world_size    = opt.world_size,
#             rank          = rank
#     )
#     session(opt,gpu)
#     dist.destroy_process_group()

if __name__ == '__main__':
    run()
    # freeze_support()

