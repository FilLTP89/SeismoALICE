import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.unic_test.pix2pix.train_pix2pix import Pix2Pix

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    pix2pix = Pix2Pix(cv)
    pix2pix.train()
  
main()