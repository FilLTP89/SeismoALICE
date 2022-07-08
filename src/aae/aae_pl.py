import os
import torch
import common.ex_common_setup as cs
from train.pl_training import STEADDataModule, ALICE
from configuration import app
from pytorch_lightning import Trainer


if __name__ == "__main__":
    cv = cs.setup()
    globals().update(cv)
    globals().update(opt.__dict__)
    NUM_WORKERS = int(os.cpu_count() / 2)
    AVAIL_GPUS = min(1, torch.cuda.device_count())


    stead = STEADDataModule(num_workers = opt.workers , opt = opt)
    model = ALICE(opt = opt, stead=stead)
    trainer =  Trainer( fast_dev_run=False,
                        gpus=AVAIL_GPUS, 
                        accelerator="dp",
                        max_epochs = opt.niter,
                        default_root_dir = opt.root_checkpoint,
                        progress_bar_refresh_rate=20)
    trainer.fit(model, stead)