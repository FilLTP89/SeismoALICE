import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.simple_test.alice.train_alice import ALICE

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    alice = ALICE(cv)
    alice.train()
  
main()
