import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.simple_test.wgan.train_wgan import WGAN

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    wgan = WGAN(cv)
    wgan.train()
  
main()
