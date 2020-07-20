# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Compute Sa response with Newmark method
"""
#===============================================================================
# Required modules
#===============================================================================
import numpy as np
#===============================================================================
# General informations
#===============================================================================
__author__ = "Filippo Gatti"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

# Response spectrum
def rsp(dtm,ths,vTn,zeta):
    nTn = vTn.size
    ntm = ths.size
    zeta = zeta/100.0
    
    beta = 0.25
    gamma = 0.5
    sd = np.empty([nTn],dtype=np.float64)
    sv = np.empty([nTn],dtype=np.float64)
    sa = np.empty([nTn],dtype=np.float64)
    psa = np.empty([nTn],dtype=np.float64)
    psv = np.empty([nTn],dtype=np.float64)
    
    y0 = 0.0
    yp0 = 0.0

    for T in range(0, nTn):
        y   = np.zeros((ntm,))
        yp  = np.zeros((ntm,))
        ypp = np.zeros((ntm,))
        wn = 2 * np.pi/max(vTn[T].tolist(),1.0e-3)
        y[0] = y0
        yp[0] = yp0
        ypp[0] = -ths[0] - 2.0 * wn * zeta * yp0 - wn**2 * y0
        keff = wn**2 + 1.0 / (beta * dtm**2) + gamma * 2.0 * wn * zeta / (beta * dtm)
        a1 = 1.0 / (beta * dtm**2) + gamma * 2.0 * wn * zeta / (beta * dtm)
        a2 = 1.0 / (beta * dtm) + 2.0 * wn * zeta * (gamma / beta - 1.0)
        a3 = (1.0 / (2.0 * beta) - 1.0) + 2.0 * wn * zeta * dtm * (gamma / (2.0 * beta) - 1.0)
        for t in range(0, ntm - 2):
            y[t + 1] = (-ths[t + 1] + a1 * y[t] + a2 * yp[t] + a3 * ypp[t]) / keff
            ypp[t + 1] = ypp[t] + (y[t + 1] - y[t] - dtm * yp[t] - dtm**2 * ypp[t] / 2.0) / (beta * dtm**2)
            yp[t + 1] = yp[t] + dtm * ypp[t] + dtm * gamma * (ypp[t + 1] - ypp[t])
        sd[T] = np.abs(y).max()
        sv[T] = np.abs(yp).max()
        sa[T] = np.abs(ypp).max()
        psa[T] = sd[T] * (2.0 * np.pi / vTn[T])**2
        psv[T] = sd[T] * 2.0 * np.pi / vTn[T]
    psa[0] = np.abs(ths).max()
    return sa,psa,sv,psv,sd
