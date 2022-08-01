import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.unic_test.pix2pix.exp_unitary_generator_pix2pix import ExpPix2PixGenerator

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    exp_generator_pix2pix = ExpPix2PixGenerator(cv)
    exp_generator_pix2pix.train()

main()