# -*- coding: utf-8 -*-
#!/usr/bin/env python3
u'''Extract STEAD with mpi4py and h5py parallel'''
u'''Required modules'''
import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI
import database.STEAD2pthMPI as s2p

u'''General informations'''
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

def main():
    """
    Start parallel process
    """
    MPI.Init()
    comm = MPI.COMM_WORLD               # Get communicator
    size = MPI.COMM_WORLD.Get_size()    # Get size of communicator
    rank = MPI.COMM_WORLD.Get_rank()    # Get the current rank
    hostname = MPI.Get_processor_name() # Get the hostname
    
    s2p.to_pth(comm,size,rank)
    MPI.Finalize()
    
if __name__=="__main__":
	main()