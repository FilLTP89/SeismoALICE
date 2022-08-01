import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.simple_test.alice.exp_unitary_discriminator_alice import ExpALICEDiscriminator

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    exp_discriminator_alice = ExpALICEDiscriminator(cv)
    exp_discriminator_alice.train()
  
main()
