
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
from optuna.samplers import TPESampler
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
import train.train_unic_hyper_parameter_broadband_v3 as aat


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
    # In main function we prepare the optuna object to tune the hyper parameters 
    # We also save all values of test in a sql database. In case of we want to 
    # checkout those parameters? we could use sql command
    # This the sql command to launch in a SQL environment:
    #               SELECT trial_params.trial_id ,trial_params.param_name, 
    #                       trial_params.param_value, trial_values.value 
    #               FROM trial_params JOIN trial_values 
    #               ON trial_params.trial_id == trial_values.trial_id 
    #               ORDER BY trial_values.value 

    # let's load the global variable to launch to program
    cv = cs.setup()
    locals().update(cv)

    # Defining the handler to log files
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name   = "broadband-classic-zyy16-zxy8-tuning-6"
    study_dir    = "tuner/broadband/test-13/"
    storage_name = "sqlite:///{}{}.db".format(study_dir,study_name)

    # let's creat the study case of hyper parameter tuning ...
    study = optuna.create_study(study_name=study_name, storage=storage_name,\
            direction="minimize",load_if_exists=True,sampler=TPESampler())
    study.optimize(partial(track, cv, study_dir), n_trials=50, timeout=60000)

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

def track(cv,study_dir,trial):
    DCA =  aat.trainer(trial=trial,cv=cv,study_dir=study_dir)
    accuracy = DCA.train_unique() 
    return accuracy

main()



