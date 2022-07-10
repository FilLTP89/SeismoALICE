import warnings
warnings.filterwarnings("ignore")
import common.ex_common_setup as cs
from test.simple_test.wgan.train_wgan import WGAN
# from test.simple_test.alice import ALICE
def main():
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)

    wgan = WGAN(cv)
    wgan.train()

    # alice = ALICE(cv)
    # alice.train()
  
main()
