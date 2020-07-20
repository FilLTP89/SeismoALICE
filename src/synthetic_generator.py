# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Generate synthetics accelerograms
Sabetta-Pugliese from original fortran program of Sabetta F. and Pugliese A.'''

u'''Required modules'''
import numpy as np
import scipy as sp
from scipy.stats import lognorm

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

S1d = {'0':0,'1':1,'2':0}
S2d = {'0':0,'1':0,'2':1}

class sp(object):
    def __init__(self,md,wd):
        self.md = md
        self.wd = wd
    def common(self,mw,dep,scc,sst,dtm):
        self.S1 = S1d[str(scc)]
        self.S2 = S2d[str(scc)]
        
        self.DV = 10**(-0.783 + 0.193*mw + \
                       0.208*np.log10((dep**2 + 5.1**2)**(0.5)) -\
                       0.133*self.S1 + 0.138*self.S2 + 0.247*sst)
        
        self.Ia = 10**( 0.729 + 0.911*mw - \
                        1.818*np.log10((dep**2 + 5.3**2)**(0.5)) +\
                        0.244*self.S1 + 0.139*self.S2 + 0.397*sst)
        u'''Time delay'''
        self.T1 = dep/7.
        u'''Other coefficients'''
        self.T2 = self.T1 + 0.5*self.DV
        self.T3 = self.T1 + 2.5*self.DV
        self.TFc = self.T2 - 3.5 - dep/50.
        self.T_cost = self.T2 + 0.5*self.DV
        u'''Duration'''
        self.tot_dur = 1.3*self.T3
        self.T4 = self.tot_dur - self.T1
        self.T_fond = self.T4/3.
        self.fo = 1./self.T_fond
        
        u'''Time vector'''
        self.vtm   = np.arange(0,self.tot_dur,dtm)
        self.ntm   = self.vtm.size
        self.t_val = self.vtm - self.TFc
    
        for i in range(self.ntm):
            if self.t_val[i] < 1:
                self.t_val[i] = 1
            if self.vtm[i] > self.T_cost:
                self.t_val[i] = self.t_val[i-1]
        u'''Nyquist frequency'''
        self.fNy = 1./2./dtm
        u'''Frequency vector'''
        self.vfr   = np.arange(self.fo,self.fNy,self.fo)
        self.nfr   = self.vfr.size
        self.ind_f = np.arange(1,self.nfr+1,1)
        
        u'''statistics - (NB: sqm_Pa.=2.5 in [Sabetta,Pugliese-1996] or =3'''
        self.sqm_Pa = np.log(self.T3/self.T2)/3.
        self.med_Pa = np.log(self.T2) + self.sqm_Pa**2
        
        u'''empirical regression for Fc [Hz] = central frequency'''
        self.Fc = np.exp(3.4 - 0.35*np.log(self.t_val) - 0.218*mw - 0.15*self.S2)
        
        u'''empirical regression for the ratio Fb/Fc ratio (frequency
        bandwidth)'''
        self.Fb_Fc = 0.44 + 0.07*mw - 0.08*self.S1 + 0.03*self.S2
        
        u'''statistics'''
        self.delta   = np.sqrt(np.log(1.+self.Fb_Fc**2))
        self.ln_beta = np.log(self.Fc) - 0.5*self.delta**2
             
    def generate(self,m):
        globals().update(self.md)
        
        self.ths = [] 
        
        if mw.size>1:
            for t in range(m):
                self.common(mw[t],dep[t],scc,sst,dtm)
                u'''Pa(t)'''
                Pa = self.Ia*lognorm(self.med_Pa,self.sqm_Pa).pdf(self.vtm)
                u'''GENERATE SYNTHETIC ACCELEROGRAMS'''
                tha = np.zeros((self.wd,))
                R = np.random.uniform(0,2.*np.pi,self.nfr)
                Ccos = -np.ones((self.nfr,))
                PS   = -np.ones((self.nfr,))
                for i in range(min(self.ntm,self.wd)):
                    u'''PS in cm*cm/s/s/s'''
                    PS  = Pa[i]/(self.ind_f*np.sqrt(2.*np.pi)*self.delta)
                    PS *= np.exp(-(np.log(self.vfr)-self.ln_beta[i])**2./(2.*self.delta**2))
                    u'''Ccos in cm/s/s'''
                    Ccos = np.sqrt(2.*PS)*\
                        np.cos(2.*np.pi*self.vfr*self.vtm[i] + R)
                    u'''acc in cm/s/s'''
                    tha[i] = Ccos.sum()
                u'''scaling'''
                self.ths.append(tha*scl)
        else:
            self.common(mw,dep,scc,sst,dtm)
            for t in range(m):
                u'''Pa(t)'''
                Pa = self.Ia*lognorm(self.med_Pa,self.sqm_Pa).pdf(self.vtm)
                u'''GENERATE SYNTHETIC ACCELEROGRAMS'''
                tha = np.zeros((self.wd,))
                R = np.random.uniform(0,2.*np.pi,self.nfr)
                Ccos = -np.ones((self.nfr,))
                PS   = -np.ones((self.nfr,))
                for i in range(min(self.ntm,self.wd)):
                    u'''PS in cm*cm/s/s/s'''
                    PS  = Pa[i]/(self.ind_f*np.sqrt(2.*np.pi)*self.delta)
                    PS *= np.exp(-(np.log(self.vfr)-self.ln_beta[i])**2./(2.*self.delta**2))
                    u'''Ccos in cm/s/s'''
                    Ccos = np.sqrt(2.*PS)*\
                        np.cos(2.*np.pi*self.vfr*self.vtm[i] + R)
                    u'''acc in cm/s/s'''
                    tha[i] = Ccos.sum()
                u'''scaling'''
                self.ths.append(tha*scl)
        return self.ths
    
    def __call__(self,md):
        self.md = md
        self.common()
        
    def get_tha(self):
        return self.ths
    
    def plot(self):
        import matplotlib as mpl
        #mpl.use('Agg')
        import matplotlib.pyplot as plt
        mpl.style.use('seaborn')
        
        plt.figure()
        hnd = []
        
        for tha in self.ths:
            h = plt.plot(self.vtm,tha)
            hnd.append(h)
#         plt.savefig(os.path.join(outf,'sdaae_losses.eps'),\
#                     bbox_inches='tight',dpi=500)
#         plt.savefig(os.path.join(outf,'sdaee_losses.png'),\
#                     bbox_inches='tight',dpi=200)
        plt.show()
if __name__=='__main__':
    md = {'mw':4.,'dep':10.,'scc':0,\
          'sst':1,'dtm':0.01,'scl':1}
    
    tha = sp(md)
    tha.generate(10)
    tha.plot()
