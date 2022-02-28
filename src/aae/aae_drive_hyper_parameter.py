
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Train and Test AAE'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
import profiling.profile_support
import optuna
import torch
import logging
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from optuna.trial import TrialState
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
import train.train_unic_hyper_parameter_combined_v2 as aat


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



def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name   = "unic-classic-zyy8-zxy8-tuning"
    study_dir    = "tuner/broadband/test-2/"
    storage_name = "sqlite:///{}{}.db".format(study_dir,study_name)

    study = optuna.create_study(study_name=study_name, storage=storage_name,\
             direction="minimize",load_if_exists=True)
    study.optimize(partial(track, cv, study_dir), n_trials=30, timeout=60000)

    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("  Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ",   len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("  Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    # torch.save(study,"./runs_both/broadband/tuning/debug/tuner/tuning.pkl")
    # df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete','system_attrs'], axis=1)
    # df.head(2)

def track(cv,study_dir,trial):
    DCA =  aat.trainer(trial=trial,cv=cv,study_dir=study_dir)
    # try:    
    accuracy = DCA.train_unique()
    # except:
    #     print("nan accuracy ... !")
    #     return 100000.
    return accuracy

main()



