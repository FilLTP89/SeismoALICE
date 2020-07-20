# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from matplotlib.pyplot import xscale
u'''Plot STEAD recording database'''
u'''Inspired by:
    https://github.com/smousavi05/STEAD/blob/master/STEAD_DEMO_2.ipynb
'''
u'''Required modules'''
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('text', usetex=True)
import argparse
from os.path import join as osj
import pandas as pd
import numpy as np

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2019, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

class STEAD(object):
    def __init__(self,**kwargs):
        super(STEAD,self).__init__()
        self.__call__(**kwargs)
        
    def __call__(self,**kwargs):
        self.__dict__.update(**kwargs)
        self.setup()
    
    def setup(self):
        self._md = pd.read_csv(osj(self.fld,'metadata_{}.csv'.format(self.version)))
        self._md = self._md[(self._md.trace_category==self.type) & 
                            (self._md.source_magnitude>=self.mw[0]) &
                            (self._md.source_magnitude<=self.mw[1]) &
                            (self._md.source_distance_km>=self.rhyp[0]) &
                            (self._md.source_distance_km<=self.rhyp[1])]
        self._md["source_depth_km"] = self._md["source_depth_km"].astype(np.float_)
        self._md = self._md[(self._md.source_depth_km>=self.dpt[0]) &
                            (self._md.source_depth_km<=self.dpt[1])]
    def get_md(self):
        self._md.head()
        self._md.info()
        self._md.describe()
        return self._md
    
    def apply_actions(self):
        k=self.keys
        if "map_eqk_to_csv" in self.actions:
            if "source_longitude" not in self.keys:
                k.append("source_longitude")
            if "source_latitude" not in self.keys:
                k.append("source_latitude")
            md=self._md[(self._md.trace_category=="earthquake_local") & 
                        (self._md.source_magnitude>4.0)][k]
            md.to_csv(osj(self.fld,"map_eqk.csv"))
        if "MRD_eqk_plot" in self.actions:
            X=self._md["source_distance_km"].values
            Y=self._md["source_magnitude"].values          
            Z=self._md["source_depth_km"].values
            plt.rcParams["figure.figsize"] = (6,4)
            cmap = plt.get_cmap('magma_r')
            im=plt.scatter(X,Y,c=Z,alpha=0.7,cmap=cmap,s=5.*Z)
            plt.xlim(1.,1000.)
            plt.ylabel("$\mathbf{M_{W} [1]}$",fontsize=14)
            plt.xlabel("$\mathbf{R_{hyp} [km]}$",fontsize=14)
            plt.gca().set_xscale('log')
            cbar=plt.colorbar(im)
            cbar.ax.set_ylabel('$\mathbf{Depth [km]}$', rotation=270)
            plt.savefig(osj(self.fld,"MRD_eqk_scatter.eps"),bbox_inches='tight',dpi =500)
            plt.close()
            #
            self._md["source_depth_km"].plot(figsize=(6,4),kind='hist',bins = 30,logy=True,facecolor='royalblue', alpha=0.70)
#             textstr = '\n'.join((r'Max: %.1f km' % (self._md['source_depth_km'].max(),),
#                                  r'Min: %.1f km' % (self._md['source_depth_km'].min(), )))
#             props = dict(boxstyle='round', facecolor='royalblue', alpha=0.15)
#             plt.text(10, 9.*10.**3, textstr, fontsize=13, verticalalignment='top')#, bbox=props)
            plt.ylim(0,10.**4)
            plt.ylabel("$\mathbf{Log Frequency [1]}$",fontweight='bold',fontsize=14)
            plt.xlabel('$\mathbf{Depth [km]}$',fontweight='bold',fontsize=14)
            plt.grid(True)
            plt.savefig(osj(self.fld,"Dh_eqk.eps"),bbox_inches='tight',dpi = 500)
            plt.close()
            #
            self._md["source_distance_km"].plot(kind='hist',bins=4,logy=True,facecolor='royalblue', alpha=0.70)
#             textstr = '\n'.join((r'Max: %.1f km' % (self._md['source_distance_km'].max(),),
#                                  r'Min: %.1f km' % (self._md['source_distance_km'].min(), )))
#             props = dict(boxstyle='round', facecolor='royalblue', alpha=0.15)
#             plt.text(6, 120000, textstr, fontsize=13, verticalalignment='top', bbox=props)
#             rc('font', weight='bold')
            plt.ylabel("$\mathbf{Log Frequency [1]}$",fontweight='bold',fontsize=14)
            plt.xlabel('$\mathbf{R_{hyp} [km]}$',fontweight='bold',fontsize=14)
            
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(osj(self.fld,"Reph_eqk.eps"),bbox_inches='tight',dpi = 500)
            plt.close()
            #
            self._md["source_magnitude"].plot(kind='hist',bins = 10,logy=False,facecolor='royalblue', alpha=0.70)
            textstr = '\n'.join((r'Max: %.1f km' % (self._md['source_magnitude'].max(),),
                                 r'Min: %.1f km' % (self._md['source_magnitude'].min(), )))
            props = dict(boxstyle='round', facecolor='royalblue', alpha=0.15)
            plt.text(6, 120000, textstr, fontsize=13, verticalalignment='top', bbox=props)
            rc('font', weight='bold')
            plt.ylabel("Frequency",fontweight='bold',fontsize=14)
            plt.xlabel('Mw',fontweight='bold',fontsize=14)
            plt.rcParams.update({'font.size': 12})
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(osj(self.fld,"Mwh_eqk.eps"),bbox_inches='tight',dpi = 500)
def main():
    parser = argparse.ArgumentParser(prefix_chars='@')
    parser.add_argument("@@fld",type=str,\
                        default="/home/filippo/Data/Filippo/aeolus/STEAD",\
                        help="Database root folder")
    parser.add_argument("@@version",type=str,default="11_13_19",help="Database version")
    parser.add_argument("@@actions",type=str,nargs="+",default=["map_eqk_to_csv","MRD_eqk_plot"],\
                        help="Actions required")
    parser.add_argument("@@keys",type=str,nargs="+",default=["source_magnitude"],\
                        help="Actions required")
    parser.add_argument("@@mw",type=float,nargs="+",default=[4.,8.],\
                        help="Magnitude limits")
    parser.add_argument("@@rhyp",type=float,nargs="+",default=[1.,1000.],\
                        help="Hypocentral distance limits")
    parser.add_argument("@@dpt",type=float,nargs="+",default=[0.,30.],\
                        help="Hypocentral distance limits")
    parser.add_argument("@@type",type=str,default="earthquake_local",\
                        help="Hypocentral distance limits")
    opt = parser.parse_args().__dict__
    
    s=STEAD(**opt)
    
    s.apply_actions()

if __name__=='__main__':
    main()