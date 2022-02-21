# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Train and Test AAE'''
u'''Required modules'''
# import warnings
# warnings.filterwarnings("ignore")
# from profile.profile_support import profile
import os
import torch
import concurrent.futures
import common.ex_common_setup as cs
import train.ddp.trainer_unic_broadband_ddp as aat
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.distributed as dist
from   configuration import app
import subprocess

# import submitit, random, sys
import GPUtil
u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2019, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

# def start_processes():
#     nodes   = int(os.environ['SLURM_NNODES'])
#     process = []
# i    for rank in range(nodes):
#         p = Process(target=run, args=(rank,))
#         p.start()
#         process.append(p)

#     for p in process:
#         p.join()

#     # mp.set_start_method('spawn')
#     # mp.spawn(run, nprocs=nodes, args=None, join=True)
#     app.logger.info("process finished ..")

def run():
    cv = cs.setup()
    globals().update(cv)

    globals().update(opt.__dict__)

    mp.set_start_method('forkserver', force=True)
    args = [cv, aat]

    app.logger.info("Distributed Data Parallel ")
    app.logger.info(f"Calculation will be done on {opt.nodes} device(s) and {torch.cuda.device_count()} GPU(s)")
    app.logger.info(f"World_size : {opt.world_size}")
    

    # # opt.rank            = int(os.environ['SLURM_PROCID'])
    # opt.local_rank      = rank
    # opt.cpus_per_task   = int(os.environ['SLURM_CPUS_PER_TASK'])


    n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    node_id = int(os.environ['SLURM_NODEID'])

    # local rank on the current node / global rank
    local_rank  = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size      = int(os.environ['SLURM_NTASKS'])
    n_gpu_per_node  = world_size // n_nodes

    # define master address and master port
    hostnames   = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR']   = master_addr
    os.environ['MASTER_PORT']   = str(29500)
    os.environ['WORLD_SIZE']    = str(world_size)
    os.environ['RANK']          = str(global_rank)

    # define whether this is the master process / if we are in distributed mode
    is_master = node_id == 0 and local_rank == 0
    multi_node  = n_nodes > 1
    multi_gpu   = world_size > 1

    # summary
    PREFIX = "%i - " % global_rank
    print(PREFIX + "Number of nodes: %i" % n_nodes)
    print(PREFIX + "Node ID        : %i" % node_id)
    print(PREFIX + "Local rank     : %i" % local_rank)
    print(PREFIX + "Global rank    : %i" % global_rank)
    print(PREFIX + "World size     : %i" % world_size)
    print(PREFIX + "GPUs per node  : %i" % n_gpu_per_node)
    print(PREFIX + "Master         : %s" % str(is_master))
    print(PREFIX + "Multi-node     : %s" % str(multi_node))
    print(PREFIX + "Multi-GPU      : %s" % str(multi_gpu))
    print(PREFIX + "Hostname       : %s" % socket.gethostname())

    opt.world_size  = world_size
    opt.local_rank  = local_rank
    opt.global_rank = global_rank
    
    
    assert torch.cuda.device_count() >= 2, f"Requires at least 2 GPUs to run, but got {opt.ngpu} GPUs"
    mp.spawn(session, nprocs = torch.cuda.device_count(), args=(opt,), join=True)

def session(gpu, opt):
    DCA = aat.trainer(gpu,opt)
    DCA.train()
    DCA.generate()

if __name__ == '__main__':
    run()
    # freeze_support()

