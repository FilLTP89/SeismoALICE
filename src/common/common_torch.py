from torch import tensor as ttns
from torch import int64 as ti64
from torch import from_numpy as tfnp
from torch import load as tload
from torch import save as tsave
from torch import randn as trnd
from torch import full as tfull
from torch import mean as tavg
from torch import cat as tcat
from torch import ones as o1s
from torch import ones_like as o1l
from torch import zeros as o0s
from torch import zeros_like as o0l
from torch import unsqueeze as usqz
from torch import log as tln
from torch import norm as tnrm
from torch import isnan as tnan
from numpy import abs, float32
from numpy.fft import fft

def ln0c(x):
    TINY = 1.0e-15
    return tln(x + TINY)

def tfft(ths,dtm,nfr=None):
    if nfr is None:
        fsa = tfnp(abs(fft(ths.cpu().data.numpy(),axis=-1)*dtm))
    else:
        fsa = tfnp(abs(fft(ths.cpu().data.numpy(),axis=-1,n=nfr)*dtm))
    return fsa

