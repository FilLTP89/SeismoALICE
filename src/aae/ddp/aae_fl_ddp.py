# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Train and Test AAE'''
u'''Required modules'''
# import warnings
# warnings.filterwarnings("ignore")
# from profile.profile_support import profile
import torch
import common.ex_common_setup as cs
import train.ddp.trainer_fl_ddp as aat
import torch.multiprocessing as mp
import torch.distributed as dist
from   configuration import app
u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2019, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"


def run():
    cv = cs.setup()
    globals().update(cv)

    globals().update(opt.__dict__)

    mp.set_start_method('spawn')
    args = [cv, aat]

    print("distributed data parallel ")
    assert torch.cuda.device_count() >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    mp.spawn(session, nprocs = opt.ngpu, args=(opt,), join=True)

    

def session(gpu, opt):
    DCA = aat.trainer(gpu,opt)
    DCA.train()

if __name__ == '__main__':
    run()
    # freeze_support()

