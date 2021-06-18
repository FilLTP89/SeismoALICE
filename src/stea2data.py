# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Train and Test AAE'''
u'''Required modules'''
import warnings
warnings.filterwarnings("ignore")
from profile_support import profile
import ex_common_setup as cs
import trainer_bbfl as aat

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
@profile

@profile
def main():
    #print("aae_drive_bbfl ...")
    u'''[SETUP] common variables/datasets'''
    cv = cs.setup()
    locals().update(cv)
    
main()
