import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.simple_test.alice.exp_unitary_generator_alice import ExpALICEGenerator

def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
    u'''[TRAIN] on datasets with WGAN strategy'''
    exp_generator_alice = ExpALICEGenerator(cv)
    exp_generator_alice.train()
  
main()
