import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.unic_test.pix2pix.exp_unitary_discriminator import ExpPix2PixDiscriminator

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    exp_discriminator_pix2pix = ExpPix2PixDiscriminator(cv)
    exp_discriminator_pix2pix.train()
  
main()
