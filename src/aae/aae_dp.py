import common.ex_common_setup as cs
from train.trainer_bb_dp import Trainer

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2019, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"


def main():
    cv = cs.setup()
    # And therefore this latter is become accessible to the methods in this class
    globals().update(cv)
    # define as global opt and passing it as a dictonnary here
    globals().update(opt.__dict__)
    # breakpoint()
    DCA = Trainer(cv, opt)
    u'''[TRAIN] neural networks'''
    DCA.train(opt)

if __name__ == "__main__":
    main()

