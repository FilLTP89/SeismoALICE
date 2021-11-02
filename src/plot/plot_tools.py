# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from matplotlib.pyplot import yscale
u'''Plot tools for nn'''

u'''Required modules'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors
import pandas as pd
import seaborn as sns
import os
import numpy as np
from common.common_nn import Variable, zcat
from tools.generate_noise import latent_resampling, noise_generator
from common.common_torch import tfft, trnd, tfnp, tnan, o0s
from database.database_sae import arias_intensity
import torch
from obspy.signal.tf_misfit import plot_tf_gofs, eg,pg
from obspy.signal.tf_misfit import eg, pg
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_gaussian_quantiles
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import pdb

from tools.gmm import GaussianMixture 
from database.toyset import Toyset
from configuration import app

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

mpl.style.use('seaborn')
plt.rcParams["figure.figsize"] = (10, 7.5)
# app.RNDM_ARGS = {'mean': 0, 'std': 1.0}

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.7):
    k=0
    for name, df_group in df.groupby(col_k):
        k+=1
    plt.figure(figsize=(10,6), dpi= 500)
    sns.set_palette("bright",k)

    def colored_scatter(x, y, c=None, edgecolor='black', linewidth=0.8):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['edgecolor']=edgecolor
            kwargs['linewidth']=linewidth
            plt.scatter(*args, **kwargs)

        return scatter
     
    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        hax=sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,kde=False,
            color=color,norm_hist=True,
        )
        hay=sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,kde=False,
            color=color,norm_hist=True,
            vertical=True
        )
        hax.set_xticks(list(np.linspace(0,10,11)))
        hay.set_yticks(list(np.linspace(0,10,11)))
    ## Do also global Hist:
    g.ax_joint.set_xticks(list(np.linspace(0,10,11)))
    g.ax_joint.set_yticks(list(np.linspace(0,10,11)))
    plt.legend(legends)
    plt.xlabel(r'$EG$')
    plt.ylabel(r'$PG$')


def plot_loss_explicit(losses,key, niter,outf="./imgs"):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    # plt.ticklabel_format(style='plain', axis='x', useOffset=False)

    # opening the csv file in 'w+' mode
    

    path =  outf+"/"+key+".txt"
    with open(path, 'w') as f:
        for item in losses:
            f.write("%s\n" % item)
    import pdb
    # pdb.set_trace()

    fig, ax = plt.subplots()
    nx = len(losses)
    ax.plot(losses, label = "losses[{0}]".format(key))
    # ax.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel("iteration")
    plt.ylabel("losses")
    plt.legend(loc="best")
    # plt.title("Evaluation of loss for {0}".format(key))
    ax.set_xticks([ i for i in range(0, nx, nx//10)])
    # ax2 = ax.twiny()
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xbound(ax.get_xbound())
    # ax2.set_xticklabels([x//41 for x in ax.get_xticks()])
    fig.subplots_adjust(top=0.85)
    fig.savefig(os.path.join(outf,"{0}.png".format(key)),format="png", bbox_inches='tight',dpi = 500)
    print("saving the {0} plot ...".format(key))
    plt.close()




def plot_loss_dict(losses,nb,title='loss',outf='./imgs'):
    sns.set(style="whitegrid")
    hfg,hax0 = plt.subplots(1,1,figsize=(6,6))
    miniter = 10000 
    min_iter = {'key':0}
    for k,v in losses.items():
        temp=min(miniter,(len(v)-1)//nb)
        min_iter[k] = temp

    min_iter.pop('key')
    loss_name = []
    for k,v in losses.items():
        if(min_iter[k]!=0):
            v=np.array(v[1:]).reshape(min_iter[k],nb,-1).mean(axis=-1).mean(axis=1)
            losses[k]=v
            loss_name.append(k)

    
    Dloss = {}
    # Dloss[r"$l^D(\mathbf{z^,},\mathbf{z},\mathbf{y},\mathbf{x})$"] = losses[r'$l^D$']
    # Dloss[r"$l^G(\mathbf{z^,},\mathbf{z},\mathbf{y},\mathbf{x})$"] = losses[r'$l^G$']
    # Dloss[r"$l_{R1}(\mathbf{y})$"] = losses[r'$l_{R1-y}$']
    # Dloss[r"$l_{R1}(\mathbf{x})$"] = losses[r'$l_{R1-x}$']

    Dloss[r"$l^D(\mathbf{z^,},\mathbf{z},\mathbf{y},\mathbf{x})$"] = losses[loss_name[0]]
    Dloss[r"$l^G(\mathbf{z^,},\mathbf{z},\mathbf{y},\mathbf{x})$"] = losses[loss_name[1]]
    Dloss[r"$l_{R1}(\mathbf{y})$"] = losses[loss_name[2]]
    Dloss[r"$l_{R1}(\mathbf{x})$"] = losses[loss_name[3]]

    clr = sns.color_palette("coolwarm",len(Dloss.keys()))
    i=0
    for k,v in Dloss.items():
        hax0.plot(range(v.size),v,linewidth=2,label=r"{}".format(k),color=clr[i])
        i+=1

    hax0.set_xlim(0,v.size)
    hax0.set_xticks(np.arange(0,1000*int(v.size//1000),1000))
    hax0.set_xlabel(r'$n_{epochs}$',fontsize=15,fontweight='bold')
    hax0.set_ylabel(r'$L_S [1]$',fontsize=15,fontweight='bold')
    #hax0.set_yscale('log')
    hax0.legend()
    plt.savefig(os.path.join(outf,title+'.eps'),\
                bbox_inches='tight',dpi=500)
    plt.savefig(os.path.join(outf,title+'.png'),\
                bbox_inches='tight',dpi=500)
    hax0.set_ylim(-0.1,0.1)
    plt.savefig(os.path.join(outf,title+'_zoom.eps'),\
                bbox_inches='tight',dpi=500)
    plt.savefig(os.path.join(outf,title+'_zoom.png'),\
                bbox_inches='tight',dpi=500)

    plt.close()
    print('loss functions done ...',outf)
    return

def plot_compare_ann2bb(Qec,Pfc,Qdc,Pdc,Fhz,Ghz,dev,vtm,trn_set,pfx='hybrid',outf='./imgs'):
    Qec.to(dev),Pdc.to(dev),Ghz.to(dev)
    Qec.eval(),Pdc.eval(),Ghz.eval()
    cnt=0
    sns.set(style="whitegrid")
    clr = sns.color_palette('tab10',5)
    for _,batch in enumerate(trn_set):
        #_,xt_data,zt_data,_,_,_,_ = batch
        xd_data,xf_data,zd_data,zf_data,_,_,_,xp_data,xs_data,xm_data = batch
        Xd = Variable(xd_data).to(dev)
        Xf = Variable(xf_data).to(dev)
        Xp = Variable(xp_data).to(dev)
        Xs = Variable(xs_data).to(dev)
        Xm = Variable(xm_data).to(dev)
        zd = Variable(zd_data).to(dev)
        zf = Variable(zf_data).to(dev)
        wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
        _,wnzc,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        _,wnzb,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)

        Xf_inp = zcat(Xf,wnxf)
        zff = Qec(Xf_inp)
        zdr = Ghz(zcat(zff,wnzb))
        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        _,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
        Xr = Pdc(zcat(zdr,wnzd))

        Xd_fsa = tfft(Xd,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xf_fsa = tfft(Xf,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xr_fsa = tfft(Xr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xm_fsa = tfft(Xm,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)
        Xf = Xf.cpu().data.numpy().copy()
        Xd = Xd.cpu().data.numpy().copy()
        Xr = Xr.cpu().data.numpy().copy()
        Xm = Xm.cpu().data.numpy().copy()
        
        for (io, ig) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
            ot,gt,rt = Xm[io, 1, :],Xr[ig, 1, :],Xd[ig,1,:]
            of,gf,ff,rf = Xm_fsa[io,1,:],Xr_fsa[ig,1,:],Xf_fsa[io,1,:],Xd_fsa[io,1,:]
            _,_,swo = arias_intensity(vtm[1]-vtm[0],ot,0.05)
            _,_,swg = arias_intensity(vtm[1]-vtm[0],gt,0.05)
            _,_,swr = arias_intensity(vtm[1]-vtm[0],rt,0.05)
            #(swr,swo,swg) = (int(swr//(vtm[1]-vtm[0])),int(swo//(vtm[1]-vtm[0])),int(swg//(vtm[1]-vtm[0])))
            (swo,swg,swr) = (np.argmax(ot),np.argmax(gt),np.argmax(rt))
            if swr>swo:
                pads=(swr-swo,0)
                ot=np.pad(ot[:-pads[0]],pads,'constant',constant_values=(0,0))
            elif swr<swo:
                pads=(0,swo-swr)
                ot=np.pad(ot[pads[1]:],pads,'constant',constant_values=(0,0))

            if swr>swg:
                pads=(swr-swg,0)
                gt=np.pad(gt[:-pads[0]],pads,'constant',constant_values=(0,0))
            elif swr<swg:
                pads=(0,swg-swr)
                gt=np.pad(gt[pads[1]:],pads,'constant',constant_values=(0,0))

            hfg,(hax0,hax2,hax1) = plt.subplots(3,1,figsize=(6,12))
            hax0.plot(vtm,rt,color='black',label='Recorded',linewidth=1.5)
            hax0.plot(vtm,ot,color=clr[0],label='ANN2BB',linewidth=1.)
            hax2.plot(vtm,rt,color='black',label='Recorded',linewidth=1.5)
            hax2.plot(vtm,gt,color=clr[3],label='this paper',linewidth=1.)

            hax1.loglog(vfr,rf,color='black',label='Recorded',linewidth=2)
            hax1.loglog(vfr,of,color=clr[0],label='ANN2BB',linewidth=1.3)
            hax1.loglog(vfr,ff,color=clr[1],label='PBS',linewidth=1)
            hax1.loglog(vfr,gf,color=clr[3],label='this paper',linewidth=1.3)
            hax0.set_xlim(0.0,int(vtm[-1]))
            hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax0.set_ylim(-1.0,1.0)
            hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax0.set_title('ANN2BB',fontsize=20,fontweight='bold')
            hax2.set_xlim(0.0,int(vtm[-1]))
            hax2.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax2.set_ylim(-1.0,1.0)
            hax2.set_yticks(np.arange(-1.0,1.25,0.25))
            hax2.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax2.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax2.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            hax0.legend(loc = "lower right",frameon=False)
            hax1.legend(loc = "lower right",frameon=False)
            plt.savefig(os.path.join(outf,"cmp_ann2bb_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"cmp_ann2bb_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                        format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
            
            plot_tf_gofs(ot,rt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                         nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                         bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                         w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                         ylim=0., clim=0.)
            plt.savefig(os.path.join(outf,"gof_cmp_ann2bb_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 300)
            plt.close()
            plot_tf_gofs(gt,rt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                         nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                         bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                         w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                         ylim=0., clim=0.)
            plt.savefig(os.path.join(outf,"gof_cmp_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 300)
            plt.close()
            cnt += 1
        cnt=0
        for (io, ig) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
            ot,gt,rt = Xf[io, 1, :],Xr[ig, 1, :],Xd[ig,1,:]
            of,gf,ff,rf = Xm_fsa[io,1,:],Xr_fsa[ig,1,:],Xf_fsa[io,1,:],Xd_fsa[io,1,:]
            _,_,swo = arias_intensity(vtm[1]-vtm[0],ot,0.05)
            _,_,swg = arias_intensity(vtm[1]-vtm[0],gt,0.05)
            _,_,swr = arias_intensity(vtm[1]-vtm[0],rt,0.05)
            #(swr,swo,swg) = (int(swr//(vtm[1]-vtm[0])),int(swo//(vtm[1]-vtm[0])),int(swg//(vtm[1]-vtm[0])))
            (swo,swg,swr) = (np.argmax(ot),np.argmax(gt),np.argmax(rt))
            if swr>swo:
                pads=(swr-swo,0)
                ot=np.pad(ot[:-pads[0]],pads,'constant',constant_values=(0,0))
            elif swr<swo:
                pads=(0,swo-swr)
                ot=np.pad(ot[pads[1]:],pads,'constant',constant_values=(0,0))

            if swr>swg:
                pads=(swr-swg,0)
                gt=np.pad(gt[:-pads[0]],pads,'constant',constant_values=(0,0))
            elif swr<swg:
                pads=(0,swg-swr)
                gt=np.pad(gt[pads[1]:],pads,'constant',constant_values=(0,0))

            hfg,(hax0,hax2,hax1) = plt.subplots(3,1,figsize=(6,12))
            hax0.plot(vtm,rt,color='black',label='Recorded',linewidth=1.5)
            hax0.plot(vtm,ot,color=clr[0],label='ANN2BB',linewidth=1.)
            hax2.plot(vtm,rt,color='black',label='Recorded',linewidth=1.5)
            hax2.plot(vtm,gt,color=clr[3],label='this paper',linewidth=1.)

            hax1.loglog(vfr,rf,color='black',label='Recorded',linewidth=2)
            hax1.loglog(vfr,of,color=clr[0],label='ANN2BB',linewidth=1.3)
            hax1.loglog(vfr,ff,color=clr[1],label='PBS',linewidth=1)
            hax1.loglog(vfr,gf,color=clr[3],label='this paper',linewidth=1.3)
            hax0.set_xlim(0.0,int(vtm[-1]))
            hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax0.set_ylim(-1.0,1.0)
            hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax0.set_title('ANN2BB',fontsize=20,fontweight='bold')
            hax2.set_xlim(0.0,int(vtm[-1]))
            hax2.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax2.set_ylim(-1.0,1.0)
            hax2.set_yticks(np.arange(-1.0,1.25,0.25))
            hax2.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax2.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax2.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            hax0.legend(loc = "lower right",frameon=False)
            hax1.legend(loc = "lower right",frameon=False)
            plt.savefig(os.path.join(outf,"cmp_ann2bb_aae_Xf_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"cmp_ann2bb_aae_Xf_%s_%u_%u.eps"%(pfx,cnt,io)),\
                        format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
            
            plot_tf_gofs(ot,rt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                         nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                         bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                         w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                         ylim=0., clim=0.)
            plt.savefig(os.path.join(outf,"gof_cmp_ann2bb_Xf_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 300)
            plt.close()
            plot_tf_gofs(gt,rt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                         nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                         bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                         w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                         ylim=0., clim=0.)
            plt.savefig(os.path.join(outf,"gof_cmp_aae_Xf_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 300)
            plt.close()
            cnt += 1

def plot_generate_hybrid_new(Qec,Pfc,Qdc,Pdc,Fhz,Ghz,dev,vtm,trn_set,opt,pfx='hybrid',outf='./imgs'):
    Qec.to(dev),Pfc.to(dev)
    Qdc.to(dev),Pdc.to(dev)
    Fhz.to(dev),Ghz.to(dev)
    Qec.eval(),Pfc.eval()
    Qdc.eval(),Pdc.eval()
    Fhz.eval(),Ghz.eval()
    cnt=0
    sns.set(style="whitegrid")
    clr = sns.color_palette('tab10',5)
    for _,batch in enumerate(trn_set):
        #_,xt_data,zt_data,_,_,_,_ = batch
        xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
        Xd = Variable(xd_data).to(dev)
        Xf = Variable(xf_data).to(dev)
        zd = Variable(zd_data).to(dev)
        zf = Variable(zf_data).to(dev)
        wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
        _,wnzc,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        _,wnzb,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)

        #Xf_inp = zcat(Xf,wnxf)
        #Xd_inp = zcat(Xd,wnxd)
        #zfr = Qec(Xf_inp)
        #zdf = Qdc(Xd_inp)
        #zdr = Ghz(zcat(zfr,wnzf))
        #zff = Fhz(zcat(zdf,wnzd))
        #_,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        #_,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
        #Xfr = Pfc(zcat(zff,wnzf))
        #Xdr = Pdc(zcat(zdr,wnzd))

        Xf_inp = zcat(Xf,wnxf)
        Xd_inp = zcat(Xd,wnxd)
        # zff = Qec(Xf_inp)
        _,zff = torch.split(Qec(Xf_inp),[opt.nzd, opt.nzf],dim=1)
        # zdd = Qdc(Xd_inp)
        zdd,_ = torch.split(Qec(Xd_inp),[opt.nzd, opt.nzf],dim=1)

        zdr = Ghz(zcat(zff,wnzb))
        zfr = Fhz(zcat(zdd,wnzd))

        _,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        _,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
        Xfr = Pfc(zcat(zfr,wnzf))
        Xdr = Pdc(zcat(zdr,wnzd))
        
        Xd_fsa = tfft(Xd,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xdr_fsa = tfft(Xdr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xf_fsa = tfft(Xf,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xfr_fsa = tfft(Xfr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        
        vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)
        Xd = Xd.cpu().data.numpy().copy()
        Xf = Xf.cpu().data.numpy().copy()
        Xdr = Xdr.cpu().data.numpy().copy()
        Xfr = Xfr.cpu().data.numpy().copy()
        
        for (io, ig) in zip(range(Xd.shape[0]),range(Xdr.shape[0])):
            ot,gt = Xd[io, 1, :],Xdr[ig, 1, :]
            of,gf,ff = Xd_fsa[io,1,:],Xdr_fsa[ig,1,:],Xf_fsa[io,1,:]

            plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                    nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                    bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                    w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                    ylim=0., clim=0.)
            print("saving file gof_dr_aae_%s_%u_%u.png"%(pfx,cnt,io))
            
            plt.savefig(os.path.join(outf,"gof_dr_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                    bbox_inches='tight',dpi = 300)
            #plt.savefig(os.path.join(outf,"gof_r_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                    #            format='eps',bbox_inches='tight',dpi = 300)
            plt.close()
            
            hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
            hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{y}$',linewidth=1.2)
            hax0.plot(vtm,gt,color=clr[3],label=r'$G_y(G_z(F_x(\mathbf{x})))$',linewidth=1.2)
            hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{y}$',linewidth=2)
            hax1.loglog(vfr,ff,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
            hax1.loglog(vfr,gf,color=clr[3],label=r'$G_y(G_z(F_x(\mathbf{x})))$',linewidth=2)
            hax0.set_xlim(0.0,int(vtm[-1]))
            hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax0.set_ylim(-1.0,1.0)
            hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax0.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            hax0.legend(loc = "lower right",frameon=False)
            hax1.legend(loc = "lower right",frameon=False)
            plt.savefig(os.path.join(outf,"res_dr_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"res_dr_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                        format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
            cnt += 1
        
        for (io, ig) in zip(range(Xf.shape[0]),range(Xfr.shape[0])):
            ot,gt = Xf[io, 1, :],Xfr[ig, 1, :]
            of,gf = Xf_fsa[io,1,:],Xfr_fsa[ig,1,:]

            plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                    nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                    bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                    w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                    ylim=0., clim=0.)
            plt.savefig(os.path.join(outf,"gof_fr_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                    bbox_inches='tight',dpi = 300)
            #plt.savefig(os.path.join(outf,"gof_r_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                    #            format='eps',bbox_inches='tight',dpi = 300)
            plt.close()
            
            hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
            hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{x}$',linewidth=1.2)
            hax0.plot(vtm,gt,color=clr[3],label=r'$G_x(F_z(\mathbf{y}))$',linewidth=1.2)
            hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{x}$',linewidth=2)
            hax1.loglog(vfr,gf,color=clr[3],label=r'$G_x(F_z(\mathbf{y}))$',linewidth=2)
            hax0.set_xlim(0.0,int(vtm[-1]))
            hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax0.set_ylim(-1.0,1.0)
            hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax0.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            hax0.legend(loc = "lower right",frameon=False)
            hax1.legend(loc = "lower right",frameon=False)
            plt.savefig(os.path.join(outf,"res_fr_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"res_fr_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                        format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
            cnt += 1

def plot_generate_hybrid(Qec,Pdc,Ghz,dev,vtm,trn_set,pfx='hybrid',outf='./imgs'):
    Qec.to(dev),Pdc.to(dev),Ghz.to(dev)
    Qec.eval(),Pdc.eval(),Ghz.eval()
    cnt=0
    sns.set(style="whitegrid")
    clr = sns.color_palette('tab10',5)
    for _,batch in enumerate(trn_set):
        #_,xt_data,zt_data,_,_,_,_ = batch
        xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
        Xd = Variable(xd_data).to(dev)
        Xf = Variable(xf_data).to(dev)
        zd = Variable(zd_data).to(dev)
        zf = Variable(zf_data).to(dev)
        wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
        wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
        X_inp = zcat(Xf,wnxf)
        zfr = Qec(X_inp)
        zdr = Ghz(zcat(zfr,wnzf))
        # ztr = latent_resampling(Qec(X_inp),zt.shape[1],wn1)
        z_inp = zcat(zdr,wnzd)
        Xr = Pdc(z_inp)
        Xd_fsa = tfft(Xd,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xf_fsa = tfft(Xf,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xr_fsa = tfft(Xr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)
        Xd = Xd.cpu().data.numpy().copy()
        Xr = Xr.cpu().data.numpy().copy()
        
        for (io, ig) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
            ot,gt = Xd[io, 1, :]  ,Xr[ig, 1, :]
            of,gf,ff = Xd_fsa[io,1,:],Xr_fsa[ig,1,:],Xf_fsa[io,1,:]

            plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                    nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                    bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                    w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                    ylim=0., clim=0.)
            plt.savefig(os.path.join(outf,"gof_r_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                    bbox_inches='tight',dpi = 300)
            #plt.savefig(os.path.join(outf,"gof_r_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                    #            format='eps',bbox_inches='tight',dpi = 300)
            plt.close()
            
            hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
            hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{y}$',linewidth=1.2)
            hax0.plot(vtm,gt,color=clr[3],label=r'$G_t(G_z(F_m(\mathbf{x})))$',linewidth=1.2)
            hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{y}$',linewidth=2)
            hax1.loglog(vfr,ff,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
            hax1.loglog(vfr,gf,color=clr[3],label=r'$G_t(G_z(F_m(\mathbf{x})))$',linewidth=2)
            hax0.set_xlim(0.0,int(vtm[-1]))
            hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            hax0.set_ylim(-1.0,1.0)
            hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax0.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            hax0.legend(loc = "lower right",frameon=False)
            hax1.legend(loc = "lower right",frameon=False)
            plt.savefig(os.path.join(outf,"res_r_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                        bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"res_r_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                        format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
            
            cnt += 1



# def plot_generate_unic(Fxy,Gx,Gy,dev,vtm,trn_set,pfx='hybrid',outf='./imgs'):
#     Fxy.to(dev),Pdc.to(dev),Ghz.to(dev)
#     Fxy.eval(),Gx.eval(),Gy.eval()
#     cnt=0
#     sns.set(style="whitegrid")
#     clr = sns.color_palette('tab10',5)
#     for _,batch in enumerate(trn_set):
#         #_,xt_data,zt_data,_,_,_,_ = batch
#         y,x,zd,zf,_,_,_,*other = batch
#         y  = y.to(dev)   # recorded signal
#         x  = x.to(dev)   # synthetic signal
#         zd = zd.to(dev)  # recorded signal latent space
#         zf = zf.to(dev)  # synthetic signal latent space

#         wny,wnzd,wn1 = noise_generator(y.shape,zd.shape,device,app.RNDM_ARGS)
#         wnx,wnzf,wn1 = noise_generator(x.shape,zf.shape,device,app.RNDM_ARGS)

#         y_inp  = zcat(y,wny)
#         x_inp  = zcat(x,wnx)
#         zd_inp = zcat(zd,wnzd)
#         zf_inp = zcat(zf,wnzf)

#         y_gen  = Gy(zd_inp)
#         x_gen  = Gx(zf_inp)
#         zd_gen = F_(y_inp)[:,:zd.shape[1],:]
#         zf_gen = F_(x_inp)[:,zd.shape[1]:,:]


#         X_fsa = tfft(x,vtm[1]-vtm[0]).cpu().data.numpy().copy()
#         Y_fsa = tfft(y,vtm[1]-vtm[0]).cpu().data.numpy().copy()

#         Xr_dsa = tfft(x_gen,vtm[1]-vtm[0]).cpu().data.numpy().copy()
#         Yr_fsa = tfft(y_gen,vtm[1]-vtm[0]).cpu().data.numpy().copy()

#         zr_dsa = tfft(zd_gen,vtm[1]-vtm[0]).cpu().data.numpy().copy()
#         zr_fsa = tfft(zf_gen,vtm[1]-vtm[0]).cpu().data.numpy().copy()


#         vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)

#         xd  = x_inp.cpu().data.numpy().copy()
#         xr  = x_gen.cpu().data.numpy().copy()

#         yd  = y_inp.cpu().data.numpy().copy()
#         yr  = y_gen.cpu().data.numpy().copy()

#         zd  = zd_inp.cpu().data.numpy().copy()
#         zdr = zd_gen.cpu().data.numpy().copy()

#         zf  = zf_inp.cpu().data.numpy().copy()
#         zfr = zf_gen.cpu().data.numpy().copy()

        
#         for (io, ig) in zip(range(x.shape[0]),range(x_gen.shape[0])):
#             # broadband signal parts
#             ot,gt = x[io, 1, :],  x_gen[ig, 1, :]
#             of,gf = X_fsa[io,1,:], Xr_fsa[ig,1,:]

#             plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
#                     nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
#                     bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
#                     w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
#                     ylim=0., clim=0.)
#             plt.savefig(os.path.join(outf,"gof_fr_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
#                     bbox_inches='tight',dpi = 300)
#             #plt.savefig(os.path.join(outf,"gof_r_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
#                     #            format='eps',bbox_inches='tight',dpi = 300)
#             plt.close()
            
#             hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))

#             # ploting original and generated values for filtered signals
#             hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{x}$',linewidth=1.2)
#             hax0.plot(vtm,gt,color=clr[3],label=r'$G_x((\mathbf{x}))$',linewidth=1.2)

#             # ploting original and generated values for broadband signals
#             hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{x}$',linewidth=2)
#             # hax1.loglog(vfr,ff,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
#             hax1.loglog(vfr,gf,color=clr[3],label=r'$G_x((\mathbf{x}))$',linewidth=2)


#             hax0.set_xlim(0.0,int(vtm[-1]))
#             hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
#             hax0.set_ylim(-1.0,1.0)
#             hax0.set_yticks(np.arange(-1.0,1.25,0.25))
#             hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
#             hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
#             hax0.set_title('DC-ALICE',fontsize=20,fontweight='bold')
#             hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
#             hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
#             hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
#             hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
#             hax0.legend(loc = "lower right",frameon=False)
#             hax1.legend(loc = "lower right",frameon=False)
#             plt.savefig(os.path.join(outf,"res_fr_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
#                         bbox_inches='tight',dpi = 500)
#             plt.savefig(os.path.join(outf,"res_fr_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
#                         format='eps',bbox_inches='tight',dpi = 500)
#             plt.close()
#             cnt += 1
#         cnt = 0    
#         for (io, ig) in zip(range(y.shape[0]),range(y_gen.shape[0])):
#             # broadband signal parts
#             ot,gt = y[io, 1, :]  ,y_gen[ig, 1, :]
#             of,gf,ff = y_fsa[io,1,:],yr_fsa[ig,1,:]

#             plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
#                     nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
#                     bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
#                     w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
#                     ylim=0., clim=0.)
#             plt.savefig(os.path.join(outf,"gof_br_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
#                     bbox_inches='tight',dpi = 300)
#             #plt.savefig(os.path.join(outf,"gof_r_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
#                     #            format='eps',bbox_inches='tight',dpi = 300)
#             plt.close()
            
#             hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
#             hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{X}$',linewidth=1.2)
#             hax0.plot(vtm,gt,color=clr[3],label=r'$G_x((\mathbf{x}))$',linewidth=1.2)

#             hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{y}$',linewidth=2)
#             # hax1.loglog(vfr,ff,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
#             hax1.loglog(vfr,gf,color=clr[3],label=r'$G_t(G_z(F_m(\mathbf{x})))$',linewidth=2)

#             hax0.set_xlim(0.0,int(vtm[-1]))
#             hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
#             hax0.set_ylim(-1.0,1.0)
#             hax0.set_yticks(np.arange(-1.0,1.25,0.25))
#             hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
#             hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
#             hax0.set_title('DC-ALICE',fontsize=20,fontweight='bold')
#             hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
#             hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
#             hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
#             hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
#             hax0.legend(loc = "lower right",frameon=False)
#             hax1.legend(loc = "lower right",frameon=False)
#             plt.savefig(os.path.join(outf,"res_br_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
#                         bbox_inches='tight',dpi = 500)
#             plt.savefig(os.path.join(outf,"res_br_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
#                         format='eps',bbox_inches='tight',dpi = 500)
#             plt.close()
            
#             cnt += 1

            # filtered signals


def plot_error(error, outf):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    nx = len(error[0])
    # plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    fig, ax = plt.subplots()
    for key in error :
        ax.plot(error[key], label = "batch #{0}".format(key))

        # ax.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel("Epochs")
    plt.ylabel("Mean Error[%]")
    plt.legend(loc="best")
    plt.title("Error with MSELoss - squared L2 norm")
    ax.set_xticks([ i for i in range(0,nx, nx//5)])
    fig.savefig(os.path.join(outf,"error.png"),format="png",\
                            bbox_inches='tight',dpi = 500)
    plt.close()

     
def plot_generate_classic(tag, Qec, Pdc, trn_set, opt=None, vtm = None, pfx='trial',outf='./imgs'):
    #Qec.to(dev),Pdc.to(dev)
    Qec.eval(),Pdc.eval()
    Qec.to(dtype =torch.float64)
    Pdc.to(dtype =torch.float64)
    dev = app.DEVICE
    cnt=0
    EG = []
    PG = []
    sns.set(style="whitegrid")
    # pdb.set_trace()
    # clr = sns.color_palette('Paired',5)
    clr = ['black', 'blue','red', 'red']
    if opt is not None:
        vtm = torch.load(os.path.join(opt.dataroot,'vtm.pth'))
    if tag=='broadband':
        # pass
        for _,batch in enumerate(trn_set):
            #_,xt_data,zt_data,_,_,_,_ = batch
            print("Plotting signals ...")
            xt_data,xf_data,zt_data,*other = batch
            # xt_data,zt_data,*other = batch
            Xt = Variable(xt_data).to(dev, dtype =torch.float64)
            Xf = Variable(xf_data).to(dev, dtype =torch.float64)
            zt = Variable(zt_data).to(dev, dtype =torch.float64)
            wnx,wnz,wn1 = noise_generator(Xt.shape,zt.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xt,wnx.to(dev))

            # ztr = Qec(X_inp).to(dev)

            # breakpoint()
            # ztr = Qec(X_inp)[0] if 'unique' in pfx else Qec(X_inp)
            if 'unique' in pfx:
                zy,zdf_gen,*other=  Qec(X_inp)
                wn = torch.empty(*zy.shape).normal_(**app.RNDM_ARGS).to(dev)
                ztr = zcat(zdf_gen,wn)
            else:
                ztr = Qec(X_inp)

            ztr = ztr.to(dev)
            # pdb.set_trace()
             # ztr = latent_resampling(Qec(X_inp),zt.shape[1],wn1)
            z_inp = zcat(ztr,wnz.to(dev))
            # breakpoint()
            # z_inp = zcat(ztr,torch.zeros_like(wnz).to(dev))
            # z_pre = zcat(zt,wnz.to(dev))
            Xr = Pdc(z_inp)
            #Xp = Pdc(z_pre)
            Xt_fsa = tfft(Xt,vtm[1]-vtm[0]).cpu().data.numpy().copy()
            Xf_fsa = tfft(Xf,vtm[1]-vtm[0]).cpu().data.numpy().copy()
            Xr_fsa = tfft(Xr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
            #Xp_fsa = tfft(Xp,vtm[1]-vtm[0]).cpu().data.numpy().copy()
            vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)
            Xt = Xt.cpu().data.numpy().copy()
            Xr = Xr.cpu().data.numpy().copy()
            #Xp = Xp.cpu().data.numpy().copy()
            
            for (io, ig) in zip(range(Xt.shape[0]),range(Xr.shape[0])):
                ot,gt = Xt[io, 1, :]  ,Xr[ig, 1, :]
                of,gf,ff = Xt_fsa[io,1,:],Xr_fsa[ig,1,:],Xf_fsa[io,1,:]
                
                plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                    nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                    bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                    w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                    ylim=0., clim=0.)
                EG.append(eg(ot,gt,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                    st2_isref=True,a=10.,k=1))
                PG.append(pg(ot,gt,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                    st2_isref=True,a=10.,k=1))
                plt.savefig(os.path.join(outf,"gof_bb_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                            bbox_inches='tight',dpi = 300)
                # plt.savefig(os.path.join(outf,"gof_bb_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                #            format='eps',bbox_inches='tight',dpi = 300)
                print("saving gof_bb_aae_%s_%u_%u ... "%(pfx,cnt,io))
                plt.close()
                
                _,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
                hax0.plot(vtm,gt,color=clr[3],label=r'$G_t(zcat(F_x(\mathbf{x},N(\mathbf{0},\mathbf{I})))$',linewidth=1.2)
                # hax0.plot(vtm,gt,color=clr[3],label=r'$G_t(zcat(F_x(\mathbf{x},\mathbf{0}))$',linewidth=1.2)
                hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{y}$',linewidth=1.2)
                hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{y}$',linewidth=2)
                hax1.loglog(vfr,ff,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
                hax1.loglog(vfr,gf,color=clr[3],label=r'$G_t(cat(F_t(\mathbf{y},z_{xy}))$',linewidth=2)
                hax0.set_xlim(0.0,int(vtm[-1]))
                hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
                hax0.set_ylim(-1.0,1.0)
                hax0.set_yticks(np.arange(-1.0,1.25,0.25))
                hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
                hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
                hax0.set_title('ALICE',fontsize=20,fontweight='bold')
                hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
                hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
                hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
                hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
                hax0.legend(loc = "lower right",frameon=False)
                hax1.legend(loc = "lower right",frameon=False)
                plt.savefig(os.path.join(outf,"res_bb_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                            bbox_inches='tight',dpi = 500)
                # plt.savefig(os.path.join(outf,"res_bb_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                #             format='eps',bbox_inches='tight',dpi = 500)
                print("saving res_bb_aae_%s_%u_%u ... "%(pfx,cnt,io))
                plt.close()
                
                cnt += 1

            # for (io, ig) in zip(range(Xt.shape[0]),range(Xp.shape[0])):
            #    ot,gt = Xt[io, 1, :]  ,Xp[ig, 1, :]
            #    of,gf,ff = Xt_fsa[io,1,:],Xp_fsa[ig,1,:],Xf_fsa[io,1,:]
               
            #    plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=20.0,
            #        nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
            #        bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
            #        w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
            #        ylim=0., clim=0.)
            #    plt.savefig(os.path.join(outf,"gof_p_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
            #                bbox_inches='tight',dpi = 500)
            #    plt.savefig(os.path.join(outf,"gof_p_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
            #                format='eps',bbox_inches='tight',dpi = 500)
            #    plt.close()
            
            #    _,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
            #    hax0.plot(vtm,ot,color=clr[0],label='original',linewidth=1.2)
            #    hax0.plot(vtm,gt,color=clr[3],label='generated',linewidth=1.2)
            #    hax1.loglog(vfr,of,color=clr[0],label='original',linewidth=2)
            #    hax1.loglog(vfr,ff,color=clr[1],label='filtered',linewidth=2)
            #    hax1.loglog(vfr,gf,color=clr[3],label='generated',linewidth=2)
            #    hax0.set_xlim(0.0,int(vtm[-1]))
            #    hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            #    hax0.set_ylim(-1.0,1.0)
            #    hax0.set_yticks(np.arange(-1.0,1.25,0.25))
            #    hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            #    hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            #    hax0.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            #    hax1.set_xlim(0.1,50.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            #    hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            #    hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            #    hax0.legend(loc = "lower right",frameon=False)
            #    hax1.legend(loc = "lower right",frameon=False)
            #    plt.savefig(os.path.join(outf,"res_p_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
            #                bbox_inches='tight',dpi = 500)
            #    plt.savefig(os.path.join(outf,"res_p_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
            #                format='eps',bbox_inches='tight',dpi = 500)
            #    plt.close()
            #    cnt += 1
        print("savefig eg_pg ...")
        plot_eg_pg(EG,PG, outf,pfx)
            
    elif 'filtered' in tag:
        for _,batch in enumerate(trn_set):
            # _,xf_data,_,zf_data,_,_,_,*other = batch
            # xt_data,xf_data,zt_data,zf_data,_,_,_,*other = batch
            _,xf_data,_,zf_data,*other = batch
            # _,xf_data,zf_data,*other = batch
            # tweaked value
            Xf = Variable(xf_data).to(dev, dtype =torch.float64)
            zf = Variable(zf_data).to(dev, dtype =torch.float64)
            wnx,wnz,wn1 = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xf,wnx)
            # ztr = Qec(X_inp)
            # pdb.set_trace()
            # ztr = Qec(X_inp)[1] if 'unique' in pfx else Qec(X_inp)
            if 'unique' in pfx:
                _,zfd_gen,zff_gen = Qec(X_inp)
                ztr = zfd_gen
            else:
                ztr = Qec(X_inp)
            # ztr = latent_resampling(Qec(X_inp),zf.shape[1],wn1)
            z_inp = zcat(ztr,wnz)
            # z_inp = zcat(ztr,torch.zeros_like(wnz).to(dev))
            Xr = Pdc(z_inp)
            Xf_fsa = tfft(Xf,vtm[1]-vtm[0]).cpu().data.numpy().copy()
            Xr_fsa = tfft(Xr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
            vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)
            Xf = Xf.cpu().data.numpy().copy()
            Xr = Xr.cpu().data.numpy().copy()
            
            for (io, ig) in zip(range(Xf.shape[0]),range(Xr.shape[0])):
                ot,gt = Xf[io, 1, :]  ,Xr[ig, 1, :]
                of,gf = Xf_fsa[io,1,:],Xr_fsa[ig,1,:]
                plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=20.0,
                        nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                        bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                        w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                        ylim=0., clim=0.)
                EG.append(eg(ot,gt,dt=vtm[1]-vtm[0],fmin=0.1,fmax=20.0,nf=100,w0=6,norm='global',
                    st2_isref=True,a=10.,k=1))
                PG.append(pg(ot,gt,dt=vtm[1]-vtm[0],fmin=0.1,fmax=20.0,nf=100,w0=6,norm='global',
                    st2_isref=True,a=10.,k=1))
                plt.savefig(os.path.join(outf,"gof_fl_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                            bbox_inches='tight',dpi = 300)
                app.logger.info("saving gof_aae_%s_%u_%u ... "%(pfx,cnt,io))
                # plt.savefig(os.path.join(outf,"gof_fl_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                #            format='eps',bbox_inches='tight',dpi = 500)
                plt.close()
                hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
                hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{x}$',linewidth=1.2)
                hax0.plot(vtm,gt,color=clr[3],label=r'$G_m(F_m(\mathbf{x}))$',linewidth=1.2)
                hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{x}$',linewidth=2)
                hax1.loglog(vfr,gf,color=clr[3],label=r'$G_m(F_m(\mathbf{x}))$',linewidth=2)
                hax0.set_xlim(0.0,int(vtm[-1]))
                hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
                hax0.set_ylim(-1.0,1.0)
                hax0.set_yticks(np.arange(-1.0,1.25,0.25))
                hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
                hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
                hax0.set_title('ALICE',fontsize=20,fontweight='bold')
                hax1.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
                hax1.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
                hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
                hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
                hax0.legend(loc = "lower right",frameon=False)
                hax1.legend(loc = "lower right",frameon=False)
                plt.savefig(os.path.join(outf,"res_fl_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                            bbox_inches='tight',dpi = 500)
                # plt.savefig(os.path.join(outf,"res_fl_aae_%s_%u_%u.eps"%(pfx,cnt,io)),\
                #             format='eps',bbox_inches='tight',dpi = 500)
                plt.close()
                app.logger.info("saving res_fl_aae_%s_%u_%u ... "%(pfx,cnt,io))
                
                cnt += 1
            app.logger.info("savefig eg_pg ...")
            plot_eg_pg(EG,PG, outf,pfx)
        # plt.scatter(EG,PG,c = 'red')
        # plt.xlabel("EG")
        # plt.ylabel("PG")
        # plt.savefig(os.path.join(outf,"gof_eg_pg.png"),\
        #                 bbox_inches='tight',dpi = 300)
        # print("saving gof_eg_pg")
        # plt.close()

def plot_ohe(ohe):
    df = pd.DataFrame(np.nan,\
                      index=['D{:>d}'.format(f) for f in range(ohe.shape[0])],\
                      columns=['L{:>d}'.format(f) for f in range(ohe.shape[1])])
    for l in range(ohe.shape[1]):
        df['L{:>d}'.format(l)]=ohe[:,l]

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df, cmap=cmap)
    ax1.grid(True)
    plt.title('One Hot Encoding')
    ax1.set_xticklabels(df.columns.tolist(),fontsize=12)
    ax1.set_yticklabels(df.index.tolist(),fontsize=12)
    plt.show()
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.savefig('onehotencoding.eps',dpi=300,bbox_inches='tight')
    plt.savefig('onehotencoding.png',dpi=300,bbox_inches='tight')
    plt.close()


def produce_df(rows, columns, row_names=None, column_names=None):
    """rows is a list of lists that will be used to build a MultiIndex
    columns is a list of lists that will be used to build a MultiIndex"""
    row_index = pd.MultiIndex.from_product(rows, names=row_names)
    col_index = pd.MultiIndex.from_product(columns, names=column_names)
    return pd.DataFrame(index=row_index, columns=col_index)


def plot_gofs(tag,Fef,Gdf,Fed,Gdd,Fhz,Ghz,dev,vtm,trn_set,
        pfx={'broadband':'trial'},outf='./imgs'):
    sns.set(style="ticks")
    Fef.to(dev),Fef.eval()
    Fed.to(dev),Fed.eval()
    Fhz.to(dev),Fhz.eval()
    Gdf.to(dev),Gdf.eval()
    Gdd.to(dev),Gdd.eval()
    Ghz.to(dev),Ghz.eval()
    bsz = trn_set.batch_size
    idx = trn_set.dataset.indices.numpy()
    cht = trn_set.dataset.dataset.inpZ.shape[1]
    chf = trn_set.dataset.dataset.inpW.shape[1]
    if 'broadband' in tag:
        bst=trn_set.dataset.dataset.inpZ[idx,:,:].data.numpy()
        egpg=np.empty((len(idx),2))
        for b,batch in enumerate(trn_set):
            xd_data,_,zd_data,_,_,_,_,*other = batch
            Xd = Variable(xd_data).to(dev)
            Xd.data[np.where(tnan(Xd).cpu().data.numpy())]=0.0
            zd = Variable(zd_data).to(dev)
            wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xd,wnx)
            zdr = Fed(X_inp)
            #if tnan(zdr).any():
             #   import pdb
              #  pdb.set_trace()
            z_inp = zcat(zdr,wnz)

            Xr = Gdd(z_inp)
            Xd = Xd.cpu().data.numpy().copy()
            Xr = Xr.cpu().data.numpy().copy()

            for (io,ir) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
                st1 = Xr[ir,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                egpg[b*bsz+io,0] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()
                egpg[b*bsz+io,1] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()
        egpg_df = pd.DataFrame(egpg,columns=['EG','PG'])
        egpg_df['kind']=r"$G_y(F_y(y))$"
        multivariateGrid('EG','PG','kind',df=egpg_df)
        plt.savefig(os.path.join(outf,"egpg_%s.png"%(pfx['broadband'])),\
                    bbox_inches='tight',dpi = 500)
        # plt.savefig(os.path.join(outf,"egpg_%s.eps"%(pfx['broadband'])),\
        #             format='eps',bbox_inches='tight',dpi = 500)
        app.logger.info("plot Gy(Fy(y)) gofs")

    if 'filtered' in tag:
        bst=trn_set.dataset.dataset.inpW[idx,:,:].data.numpy()
        egpg=np.empty((len(idx),2))
        for b,batch in enumerate(trn_set):
            _,xd_data,_,zd_data,_,_,_,*other = batch
            Xd = Variable(xd_data).to(dev)
            Xd.data[np.where(tnan(Xd).cpu().data.numpy())]=0.0
            zd = Variable(zd_data).to(dev)
            wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xd,wnx)
            zdr = Fef(X_inp)
            #if tnan(zdr).any():
               # import pdb
               #pdb.set_trace()
            z_inp = zcat(zdr,wnz)

            Xr = Gdf(z_inp)
            Xd = Xd.cpu().data.numpy().copy()
            Xr = Xr.cpu().data.numpy().copy()

            for (io,ir) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
                st1 = Xr[ir,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                egpg[b*bsz+io,0] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()
                egpg[b*bsz+io,1] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()
        egpg_df = pd.DataFrame(egpg,columns=['EG','PG'])
        egpg_df['kind']=r"$G_x(F_x(x))$"
        multivariateGrid('EG','PG','kind',df=egpg_df)
        plt.savefig(os.path.join(outf,"egpg_%s.png"%(pfx['filtered'])),\
                    bbox_inches='tight',dpi = 500)
        # plt.savefig(os.path.join(outf,"egpg_%s.eps"%(pfx['filtered'])),\
        #             format='eps',bbox_inches='tight',dpi = 500)
        app.logger.info("plot Gx(Fx(x)) gofs")
    if 'hybrid' in tag:
        bst=trn_set.dataset.dataset.inpZ[idx,:,:].data.numpy()
        egpg=np.empty((len(idx),6))
        for b,batch in enumerate(trn_set):
            xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
            Xd = Variable(xd_data).to(dev)
            Xd.data[np.where(tnan(Xd).cpu().data.numpy())]=0.0
            zd = Variable(zd_data).to(dev)
            Xf = Variable(xf_data).to(dev)
            Xf.data[np.where(tnan(Xf).cpu().data.numpy())]=0.0
            zf = Variable(zf_data).to(dev)
            wnxd,wnzd,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xd,wnxd)
            zdr = Fed(X_inp)
            #if tnan(zdr).any():
                #import pdb
                #pdb.set_trace()
            z_inp = zcat(zdr,wnzd)

            Xr = Gdd(z_inp)
            Xd = Xd.cpu().data.numpy().copy()
            Xr = Xr.cpu().data.numpy().copy()
            
            _,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
            Xf_inp = zcat(Xf,wnxf)
            zff = Fef(Xf_inp)
            #if tnan(zff).any():
               # import pdb
               # pdb.set_trace()
            zdr = Ghz(zcat(zff,wnzf))
            Xdr = Gdd(zcat(zdr,wnzd))
            Xdr = Xdr.cpu().data.numpy().copy()
            Xf = Xf.cpu().data.numpy().copy()

            for (io,ir) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
                # broadband
                st1 = Xr[ir,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                egpg[b*bsz+io,0] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                egpg[b*bsz+io,1] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                # hybrid
                st1 = Xdr[ir,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                egpg[b*bsz+io,2] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                egpg[b*bsz+io,3] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                # filtered
                st1 = Xf[ir,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                egpg[b*bsz+io,4] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                egpg[b*bsz+io,5] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)

        egpg_df_bb = pd.DataFrame(egpg[:,:2],columns=['EG','PG']) ##.astype(int)
        egpg_df_hb = pd.DataFrame(egpg[:,2:4],columns=['EG','PG']) ##.astype(int)
        egpg_df_fl = pd.DataFrame(egpg[:,4:],columns=['EG','PG']) ##.astype(int)
        egpg_df_bb['kind']=r"$G_y(F_y(y))$"
        egpg_df_hb['kind']=r"$G_y(G_z(F_x(x)))$"
        egpg_df_fl['kind']=r"$x$"
        egpg_df = pd.concat([egpg_df_bb,egpg_df_hb,egpg_df_fl])
        multivariateGrid('EG','PG','kind',df=egpg_df)
        plt.savefig(os.path.join(outf,"egpg_%s.png"%(pfx['hybrid'])),\
                    bbox_inches='tight',dpi = 500)
        # plt.savefig(os.path.join(outf,"egpg_%s.eps"%(pfx['hybrid'])),\
        #             format='eps',bbox_inches='tight',dpi = 500)
        app.logger.info("plot Gy(Gz(Fx(x))) gofs")

    if 'ann2bb' in tag:
        bst=trn_set.dataset.dataset.inpZ[idx,:,:].data.numpy()
        egpg=np.empty((len(idx),6))
        for b,batch in enumerate(trn_set):
            xd_data,xf_data,zd_data,zf_data,_,_,_,xp_data,xs_data,xm_data = batch
            Xm = Variable(xm_data).to(dev)
            Xm.data[np.where(tnan(Xm).cpu().data.numpy())]=0.0
            Xd = Variable(xd_data).to(dev)
            Xd.data[np.where(tnan(Xd).cpu().data.numpy())]=0.0
            zd = Variable(zd_data).to(dev)
            Xf = Variable(xf_data).to(dev)
            Xf.data[np.where(tnan(Xf).cpu().data.numpy())]=0.0
            zf = Variable(zf_data).to(dev)

            _,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
            Xf_inp = zcat(Xf,wnxf)
            zff = Fef(Xf_inp)
            #if tnan(zff).any():
                #import pdb
                #pdb.set_trace()
            zdr = Ghz(zcat(zff,wnzf))
            Xdr = Gdd(zcat(zdr,wnzd))

            Xd = Xd.cpu().data.numpy().copy()
            Xf = Xf.cpu().data.numpy().copy()
            Xr = Xm.cpu().data.numpy().copy()
            Xdr = Xdr.cpu().data.numpy().copy()

            for (io,ir) in zip(range(Xd.shape[0]),range(Xr.shape[0])):
                # ANN2BB
                st1 = Xr[io,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                st1 = pad3d(st1,st2,vtm,style='arias')
                egpg[b*bsz+io,0] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                egpg[b*bsz+io,1] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                # paper
                st1 = Xdr[io,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                st1 = pad3d(st1,st2,vtm,style='arias')
                egpg[b*bsz+io,2] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                egpg[b*bsz+io,3] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                # Filtered
                st1 = Xf[io,:,:].squeeze()
                st2 = Xd[io,:,:].squeeze()
                st1 = pad3d(st1,st2,vtm,style='arias')
                egpg[b*bsz+io,4] = eg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)
                egpg[b*bsz+io,5] = pg(st1,st2,dt=vtm[1]-vtm[0],fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',
                                      st2_isref=True,a=10.,k=1.).mean()#.round().astype(np.int32)

        egpg_df_bb = pd.DataFrame(egpg[:,:2],columns=['EG','PG']) ##.astype(int)
        egpg_df_hb = pd.DataFrame(egpg[:,2:4],columns=['EG','PG']) ##.astype(int)
        egpg_df_fl = pd.DataFrame(egpg[:,4:],columns=['EG','PG']) ##.astype(int)
        egpg_df_bb['kind']=r"$ANN2BB$"
        egpg_df_hb['kind']=r"$G_y(G_z(F_x(x)))$"
        egpg_df_fl['kind']=r"$x$"
        egpg_df = pd.concat([egpg_df_bb,egpg_df_hb,egpg_df_fl])
        multivariateGrid('EG','PG','kind',df=egpg_df)
        plt.savefig(os.path.join(outf,"egpg_%s.png"%(pfx['ann2bb'])),\
                    bbox_inches='tight',dpi = 500)
        plt.savefig(os.path.join(outf,"egpg_%s.eps"%(pfx['ann2bb'])),\
                    format='eps',bbox_inches='tight',dpi = 500)
        print("plot Gy(Gz(Fx(x))) gofs")

def pad3d(st1t,st2,vtm,style='arias'):
    if style == 'arias':
        _,_,swgs = arias_intensity(vtm[1]-vtm[0],st1t,0.05)
        _,_,swrs = arias_intensity(vtm[1]-vtm[0],st2,0.05)
        (swrs,swgs) = ((swrs//(vtm[1]-vtm[0])).astype(int),(swgs//(vtm[1]-vtm[0])).astype(int))
    elif style == 'pga':
        (swgs,swrs) = (np.argmax(st1t,axis=-1),np.argmax(st2,axis=-1))
    st1 = np.zeros_like(st1t)
    for ix in range(3):
        swr=swrs[ix]
        swg=swgs[ix]
        if swr>swg:
            pads=(swr-swg,0)
            tt=np.pad(st1t[ix,:-pads[0]],pads,'constant',constant_values=(0,0)).squeeze()
        elif swr<swg:
            pads=(0,swg-swr)
            tt=np.pad(st1t[ix,pads[1]:],pads,'constant',constant_values=(0,0)).squeeze()
        st1[ix,:]=tt
    return st1
def plot_features(tag,Qec,Pdc,nz,dev,vtm,trn_set,pfx='trial',outf='./imgs'):
    print("Plotting features ...")
    sns.set(style="whitegrid")
    #Qec.to(dev)
    Qec.eval()
    #Pdc.to(dev)
    Pdc.eval()
    bsz = trn_set.batch_size
    idx = trn_set.dataset.indices.numpy()
    cht = trn_set.dataset.dataset.inpZ.shape[1]
    chf = trn_set.dataset.dataset.inpW.shape[1]
    vfr = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)
    if 'broadband' in tag:
        clr = sns.color_palette("Spectral",cht)
        clt = sns.color_palette("hls",4)
        #clr = sns.color_palette("bright",cht)
        bst=trn_set.dataset.dataset.inpZ[idx,:,:].data.numpy()
        bst=np.transpose(bst,(0,2,1))
        zd_tf=np.empty_like(bst)
        zg_tf=np.empty_like(bst)
        lb_tf=np.tile(np.arange(bst.shape[-1]).reshape((1,1,-1)),\
                (bst.shape[0],bst.shape[1],1)).reshape(-1,bst.shape[-1])
        for b,batch in enumerate(trn_set):
            xd_data,xf_data,zd_data,_,_,_,_,*other = batch
            Xd = Variable(xd_data).to(dev)
            zd = Variable(zd_data).to(dev)
            wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xd,wnx)
            zg = Qec(X_inp)
            
            for c in range(zd.shape[1]):
                hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(8,8))
                _,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
                zt = torch.zeros_like(zg).detach().to(dev)
                zt[:,c,:]=zg[:,c,:]
                xths = Pdc(zcat(zt,wnz))
                xfsa = tfft(xths,vtm[1]-vtm[0])
                xths = xths[0,0,:].cpu().data.numpy()
                xfsa = xfsa[0,0,:].cpu().data.numpy()
                hax0.plot(vtm,xths,color=clr[c],label=r'$ch {{{:>d}}}$'.format(c),linewidth=1.2)
                hax1.loglog(vfr,xfsa,color=clr[c],label=r'$ch {{{:>d}}}$'.format(c),linewidth=1.2)

                hax0.set_xlim(0.0,int(vtm[-1]))
                #hax0.set_ylim(-1.,1.)
                hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
                #hax0.set_yticks(np.arange(-1.0,1.25,0.25))

                hax1.set_xlim(0.1,51.)
                hax1.set_ylim(10.**-4,10.**1) 
                hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
                hax1.set_yticks(10.**np.arange(-4,2))
                
                hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
                hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
                hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
                hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')

                #hax0.legend(loc = "lower left",frameon=False,fontsize=20)
                #hax1.legend(loc = "lower left",frameon=False,fontsize=20)
                hax0.tick_params(which='major', labelsize=12)
                hax1.tick_params(which='major', labelsize=12)
                plt.savefig(os.path.join(outf,"fts_ths_%s_%u_%u.png"%(pfx,b,c)),\
                        bbox_inches='tight',dpi = 500)
                plt.savefig(os.path.join(outf,"fts_ths_%s_%u_%u.eps"%(pfx,b,c)),\
                        format='eps',bbox_inches='tight',dpi = 500)
                plt.close()
            zw = zg
            zw =  zw.to(dev)
            zg = zg.detach().cpu().data.numpy().copy()
            zd = zd.detach().cpu().data.numpy().copy()
            zd = np.nan_to_num(zd)
            zg = np.nan_to_num(zg)
            zd_tf[b*bsz:(b+1)*bsz,:,:]=np.transpose(zd,(0,2,1)) 
            zg_tf[b*bsz:(b+1)*bsz,:,:]=np.transpose(zg,(0,2,1))
            
        for c in range(1):
            c=c+15 #range(zd.shape[1]):
            hfg1,(hax3,hax4) = plt.subplots(2,1,figsize=(8,8))
            _,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            wnz.to(dev)
            wn1.to(dev)
            for w in range(1): #range(zd.shape[2]):
                w = w +1
                dwt = 2**5*(vtm[1]-vtm[0]) 
                wtm = np.arange(0.0,dwt*zw.shape[2],dwt)
                print("[!] wtm[-1]...", wtm[-1])
                xw = Pdc(zcat(zw,wnz))
                zwfsa = tfft(zw,dwt,nfr=vfr.size)
                xwfsa = tfft(xw,vtm[1]-vtm[0],nfr=vfr.size)
                hfg,hax00 = plt.subplots(3,2,figsize=(14,8))
                hax0=hax00[1][0]
                hax1=hax00[0][1]
                hax2=hax00[1][1]
                hax5=hax00[2][1]
                hax00[0][0].set_xticks([])
                hax00[0][0].set_yticks([])
                hax00[2][0].set_xticks([])
                hax00[2][0].set_yticks([])
                hax00[0][0].grid(False)
                hax00[2][0].grid(False)
                hax00[0][0].axis('off')
                hax00[2][0].axis('off')
                hax1.plot(vtm,xw[0,0,:].cpu().data.numpy(),color='black',label=r'$G_y(F_y(y))_0$',linewidth=0.8)
                hax2.plot(vtm,xw[0,1,:].cpu().data.numpy(),color='black',label=r'$G_y(F_y(y))_1$',linewidth=0.8)
                hax5.plot(vtm,xw[0,2,:].cpu().data.numpy(),color='black',label=r'$G_y(F_y(y))_2$',linewidth=0.8)
                hax0.plot(wtm,zw[0,c,:].cpu().data.numpy(),color='black',label=r'$F_y(y)_{{{:>d}}}$'.format(c),linewidth=1.2)
                #hax2.loglog(vfr,xwfsa[0,0,:].cpu().data.numpy(),color=clt[0+1],label=r'$G_y(F_y(y))_0$',linewidth=1.5)
                #hax2.loglog(vfr,xwfsa[0,1,:].cpu().data.numpy(),color=clt[0+2],label=r'$G_y(F_y(y))_1$',linewidth=1.5)
                #hax2.loglog(vfr,xwfsa[0,2,:].cpu().data.numpy(),color=clt[0+3],label=r'$G_y(F_y(y))_2$',linewidth=1.5)
                #hax2.loglog(vfr,zwfsa[0,c,:].cpu().data.numpy(),color=clt[0],label=r'$ch {{{:>d}}}$'.format(c),linewidth=2.0)
                hax0.set_xlim(0.0,round(wtm[-1]))
                hax1.set_xlim(0.0,round(wtm[-1]))
                hax2.set_xlim(0.0,round(wtm[-1]))
                hax5.set_xlim(0.0,round(wtm[-1]))
                hax0.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax1.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax2.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax5.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax1.set_ylim(-1.0,1.0)
                hax1.set_yticks(np.arange(-1.0,1.25,0.25))
                hax2.set_ylim(-1.0,1.0)
                hax2.set_yticks(np.arange(-1.0,1.25,0.25))
                hax5.set_ylim(-1.0,1.0)
                hax5.set_yticks(np.arange(-1.0,1.25,0.25))
                hax0.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax0.set_ylabel(r"$F_y(y)_i [1]$",fontsize=15,fontweight='bold')
                hax1.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax1.set_ylabel(r"$y_0 [1]$",fontsize=15,fontweight='bold')
                hax2.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax2.set_ylabel(r"$y_1 [1]$",fontsize=15,fontweight='bold')
                hax5.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax5.set_ylabel(r"$y_2 [1]$",fontsize=15,fontweight='bold')
                hax0.tick_params(which='major', labelsize=12)
                hax1.tick_params(which='major', labelsize=12)
                hax2.tick_params(which='major', labelsize=12)
                hax5.tick_params(which='major', labelsize=12)
                #hax0.set_title('Features',fontsize=20,fontweight='bold')
                #hax2.set_xlim(0.1,51.), hax2.set_xticks(np.array([0.1,1.0,10.,50.]))
                #hax2.set_ylim(10.**-4,10.**1), hax2.set_yticks(10.**np.arange(-4,2))
                #hax2.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
                #hax2.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
                plt.savefig(os.path.join(outf,"fts_zw_%s_%u_%u.png"%(pfx,b,c)),\
                        bbox_inches='tight',dpi = 500)
                plt.savefig(os.path.join(outf,"fts_zw_%s_%u_%u.eps"%(pfx,b,c)),\
                        format='eps',bbox_inches='tight',dpi = 500)
                plt.close(fig=hfg)
                
                print("[!] informations for zt and zw", zt.shape, zw.shape)
                option = 2
                if option==0:
                    zt = torch.zeros_like(zw).detach().to(dev)
                    zt[:,c,w]=zw[:,c,w]
                elif option==1:
                    zt = torch.zeros_like(zw).detach().to(dev)
                elif option==2:
                    zt = torch.ones_like(zw).detach().fill_(1.0e4).to(dev)
                    zt[:,c,w]=0.0
                xths = Pdc(zcat(zt,wnz.fill_(0.0)))
                xfsa = tfft(xths,vtm[1]-vtm[0])
                xths = xths[0,0,:].cpu().data.numpy()
                xfsa = xfsa[0,0,:].cpu().data.numpy()
                hax3.plot(vtm,xths,color='black',linewidth=1.2)
                hax4.loglog(vfr,xfsa,color='black',linewidth=2.0)
            hax3.set_xlim(0.0,int(vtm[-1]))
            hax3.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            #hax2.set_ylim(-1.0,1.0)
            #hax2.set_yticks(np.arange(-1.0,1.25,0.25))
            hax3.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax3.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax3.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax4.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax4.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax4.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax4.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            plt.savefig(os.path.join(outf,"fts_xw_%s_%u_%u.png"%(pfx,b,c)),\
                    bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"fts_xw_%s_%u_%u.eps"%(pfx,b,c)),\
                    format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
        zd_fl = zd_tf.reshape(-1,zd_tf.shape[2])
        zg_fl = zg_tf.reshape(-1,zd_tf.shape[2])
        #
        zi = ['z'+str(i) for i in range(zd_fl.shape[1])]
        zd_df = pd.DataFrame(zd_fl,columns=zi)
        zg_df = pd.DataFrame(zg_fl,columns=zi)

        Czz_zd = EmpiricalCovariance().fit(zd_fl)
        Czz_zg = EmpiricalCovariance().fit(zg_fl)

        z_correlation_matrix(zd_df,os.path.join(outf,"zd_{}".format(pfx)),\
                                                cov=Czz_zd.covariance_,prc=Czz_zd.precision_)
        z_correlation_matrix(zg_df,os.path.join(outf,"zg_{}".format(pfx)),\
                                                cov=Czz_zg.covariance_,prc=Czz_zg.precision_)
        z_histogram(zd_df,zg_df,os.path.join(outf,"histogram_{}".format(pfx))) 
        #zd_ch0_flt = zd_all[:,0,:].reshape(-1,zd_all.shape[2])
        #zg_ch0_flt = zg_all[:,0,:].reshape(-1,zd_all.shape[2])
        #zd_ch0_df = zd_df.loc[:,'z0':'z127']
        #zg_ch0_df = zg_df.loc[:,'z0':'z127']
        #cov_ch0_zd = EmpiricalCovariance().fit(zd_ch0_flt)
        #cov_ch0_zg = EmpiricalCovariance().fit(zg_ch0_flt)
        #
        #correlation_matrix(zd_ch0_df,os.path.join(outf,"zd_ch0_{}".format(pfx)),\
        #        cov=cov_ch0_zd.covariance_,prc=cov_ch0_zd.precision_)
        #correlation_matrix(zg_ch0_df,os.path.join(outf,"zg_ch0_{}".format(pfx)),\
        #        cov=cov_ch0_zg.covariance_,prc=cov_ch0_zg.precision_)

        for i,c in zip(range(len(zg_df.columns)),zg_df):
            plt.figure(figsize=(8,6))
            sns.distplot(zg_df[c],color=clr[i],label=r'$F_y(y)_{{ {{{:>d}}} }}$'.format(i),hist_kws={'alpha':.7},kde_kws={'linewidth':3})
            plt.title(r'Density Plot $F_y(y)_{{ {{{:>d}}} }}$'.format(i),fontsize=25)
            plt.xlim(-10.0,10.0)
            plt.ylim(0.0,1.0)
            plt.legend(fontsize=20) 
            plt.savefig(os.path.join(outf,"zg_hst_%s_z%u.png"%(pfx,i)),\
                    bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"zg_hts_%s_z%u.eps"%(pfx,i)),\
                    format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
        for i,c in zip(range(len(zd_df.columns)),zd_df):
            plt.figure(figsize=(8,6))
            sns.distplot(zd_df[c],color=clr[i],label=r'$z^,_{{ {{{:>d}}} }}$'.format(i),hist_kws={'alpha':.7},kde_kws={'linewidth':3})
            plt.xlim(-10.0,10.0)
            plt.ylim(0.0,1.0)
            plt.title(r'Density Plot $z^,_{{ {{{:>d}}} }}$'.format(i),fontsize=25)
            plt.legend(fontsize=20) 
            plt.savefig(os.path.join(outf,"zd_hst_%s_z%u.png"%(pfx,i)),\
                    bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"zd_hts_%s_z%u.eps"%(pfx,i)),\
                    format='eps',bbox_inches='tight',dpi = 500)
    elif 'filtered' in tag:
        clr = sns.color_palette("Spectral",chf)
        clt = sns.color_palette("hls",4)
        bst=trn_set.dataset.dataset.inpW[idx,:,:].data.numpy()
        bst=np.transpose(bst,(0,2,1))
        zd_tf=np.empty_like(bst)
        zg_tf=np.empty_like(bst)
        lb_tf=np.tile(np.arange(bst.shape[-1]).reshape((1,1,-1)),\
                (bst.shape[0],bst.shape[1],1)).reshape(-1,bst.shape[-1])
        for b,batch in enumerate(trn_set):
            # _,xd_data,_,zd_data,_,_,_,*other = batch
            xd_data,xf_data,zd_data,_,_,_,_,*other = batch
            Xd = Variable(xd_data).to(dev)
            zd = Variable(zd_data).to(dev)
            wnx,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xd,wnx)
            zg = Qec(X_inp)
            
            for c in range(zd.shape[1]):
                hfg,(hax0,hax1) = plt.subplots(2,1,figsize=(8,8))
                _,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
                zt = torch.zeros_like(zg).detach().to(dev)
                zt[:,c,:]=zg[:,c,:]
                xths = Pdc(zcat(zt,wnz))
                xfsa = tfft(xths,vtm[1]-vtm[0])
                xths = xths[0,0,:].cpu().data.numpy()
                xfsa = xfsa[0,0,:].cpu().data.numpy()
                hax0.plot(vtm,xths,color=clr[c],label=r'$ch {{{:>d}}}$'.format(c),linewidth=1.2)
                hax1.loglog(vfr,xfsa,color=clr[c],label=r'$ch {{{:>d}}}$'.format(c),linewidth=1.2)

                hax0.set_xlim(0.0,int(vtm[-1]))
                hax0.set_ylim(-0.2,0.2)
                hax0.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
                hax0.set_yticks(np.arange(-0.2,0.25,0.05))

                hax1.set_xlim(0.1,51.)
                hax1.set_ylim(10.**-4,10.**1) 
                hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
                hax1.set_yticks(10.**np.arange(-4,2))
                
                hax0.set_xlabel('t [s]',fontsize=15,fontweight='bold')
                hax0.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
                hax1.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
                hax1.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')

                #hax0.legend(loc = "lower left",frameon=False,fontsize=20)
                #hax1.legend(loc = "lower left",frameon=False,fontsize=20)
                hax0.tick_params(which='major', labelsize=12)
                hax1.tick_params(which='major', labelsize=12)
                plt.savefig(os.path.join(outf,"fts_ths_%s_%u_%u.png"%(pfx,b,c)),\
                        bbox_inches='tight',dpi = 500)
                plt.savefig(os.path.join(outf,"fts_ths_%s_%u_%u.eps"%(pfx,b,c)),\
                        format='eps',bbox_inches='tight',dpi = 500)
                plt.close()
            zw = zg
            zg = zg.detach().cpu().data.numpy().copy()
            zd = zd.detach().cpu().data.numpy().copy()
            zd = np.nan_to_num(zd)
            zg = np.nan_to_num(zg)
            zd_tf[b*bsz:(b+1)*bsz,:,:]=np.transpose(zd,(0,2,1)) 
            zg_tf[b*bsz:(b+1)*bsz,:,:]=np.transpose(zg,(0,2,1))
            
        for c in range(1):
            c=c+5 #range(zd.shape[1]):
            hfg1,(hax3,hax4) = plt.subplots(2,1,figsize=(8,8))
            _,wnz,wn1 = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            for w in range(1): #range(zd.shape[2]):
                w = w +3
                dwt = 2**5*(vtm[1]-vtm[0]) 
                wtm = np.arange(0.0,dwt*zw.shape[2],dwt)
                xw = Pdc(zcat(zw,wnz))
                zwfsa = tfft(zw,dwt,nfr=vfr.size)
                xwfsa = tfft(xw,vtm[1]-vtm[0],nfr=vfr.size)
                hfg,hax00 = plt.subplots(3,2,figsize=(14,8))
                hax0=hax00[1][0]
                hax1=hax00[0][1]
                hax2=hax00[1][1]
                hax5=hax00[2][1]
                hax00[0][0].set_xticks([])
                hax00[0][0].set_yticks([])
                hax00[2][0].set_xticks([])
                hax00[2][0].set_yticks([])
                hax00[0][0].grid(False)
                hax00[2][0].grid(False)
                hax00[0][0].axis('off')
                hax00[2][0].axis('off')
                hax1.plot(vtm,xw[0,0,:].cpu().data.numpy(),color='black',label=r'$G_x(F_x(x))_0$',linewidth=0.8)
                hax2.plot(vtm,xw[0,1,:].cpu().data.numpy(),color='black',label=r'$G_x(F_x(x))_1$',linewidth=0.8)
                hax5.plot(vtm,xw[0,2,:].cpu().data.numpy(),color='black',label=r'$G_x(F_x(x))_2$',linewidth=0.8)
                hax0.plot(wtm,zw[0,c,:].cpu().data.numpy(),color='black',label=r'$F_x(x)_{{{:>d}}}$'.format(c),linewidth=1.2)
                #hax2.loglog(vfr,xwfsa[0,0,:].cpu().data.numpy(),color=clt[0+1],label=r'$G_x(F_x(x))_0$',linewidth=1.5)
                #hax2.loglog(vfr,xwfsa[0,1,:].cpu().data.numpy(),color=clt[0+2],label=r'$G_x(F_x(x))_1$',linewidth=1.5)
                #hax2.loglog(vfr,xwfsa[0,2,:].cpu().data.numpy(),color=clt[0+3],label=r'$G_x(F_x(x))_2$',linewidth=1.5)
                #hax2.loglog(vfr,zwfsa[0,c,:].cpu().data.numpy(),color=clt[0],label=r'$ch {{{:>d}}}$'.format(c),linewidth=2.0)
                hax0.set_xlim(0.0,round(wtm[-1]))
                hax1.set_xlim(0.0,round(wtm[-1]))
                hax2.set_xlim(0.0,round(wtm[-1]))
                hax5.set_xlim(0.0,round(wtm[-1]))
                hax0.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax1.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax2.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax5.set_xticks(np.arange(0.0,round(wtm[-1])*11./10.,round(wtm[-1])/10.))
                hax1.set_ylim(-0.2,0.2)
                hax1.set_yticks(np.arange(-0.2,0.25,0.05))
                hax2.set_ylim(-0.2,0.2)
                hax2.set_yticks(np.arange(-0.2,0.25,0.05))
                hax5.set_ylim(-0.2,0.2)
                hax5.set_yticks(np.arange(-0.2,0.25,0.05))
                hax0.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax0.set_ylabel(r"$F_x(x)_i [1]$",fontsize=15,fontweight='bold')
                hax1.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax1.set_ylabel(r"$x_0 [1]$",fontsize=15,fontweight='bold')
                hax2.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax2.set_ylabel(r"$x_1 [1]$",fontsize=15,fontweight='bold')
                hax5.set_xlabel(r'$t [s]$',fontsize=15,fontweight='bold')
                hax5.set_ylabel(r"$x_2 [1]$",fontsize=15,fontweight='bold')
                hax0.tick_params(which='major', labelsize=12)
                hax1.tick_params(which='major', labelsize=12)
                hax2.tick_params(which='major', labelsize=12)
                hax5.tick_params(which='major', labelsize=12)
                #hax0.set_title('Features',fontsize=20,fontweight='bold')
                #hax2.set_xlim(0.1,51.), hax2.set_xticks(np.array([0.1,1.0,10.,50.]))
                #hax2.set_ylim(10.**-4,10.**1), hax2.set_yticks(10.**np.arange(-4,2))
                #hax2.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
                #hax2.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
 
                plt.savefig(os.path.join(outf,"fts_zw_%s_%u_%u.png"%(pfx,b,c)),\
                        bbox_inches='tight',dpi = 500)
                plt.savefig(os.path.join(outf,"fts_zw_%s_%u_%u.eps"%(pfx,b,c)),\
                        format='eps',bbox_inches='tight',dpi = 500)
                plt.close(fig=hfg)
                zt = torch.zeros_like(zw).detach().to(dev)
                zt[:,c,w]=zw[:,c,w]
                xths = Pdc(zcat(zt,wnz.fill_(0.0)))
                xfsa = tfft(xths,vtm[1]-vtm[0])
                xths = xths[0,0,:].cpu().data.numpy()
                xfsa = xfsa[0,0,:].cpu().data.numpy()
                hax3.plot(vtm,xths,color=clr[c],linewidth=1.2)
                hax4.loglog(vfr,xfsa,color=clr[c],linewidth=2.0)
            hax3.set_xlim(0.0,int(vtm[-1]))
            hax3.set_xticks(np.arange(0.0,int(vtm[-1])*11./10.,int(vtm[-1])/10.))
            #hax2.set_ylim(-1.0,1.0)
            #hax2.set_yticks(np.arange(-1.0,1.25,0.25))
            hax3.set_xlabel('t [s]',fontsize=15,fontweight='bold')
            hax3.set_ylabel('a(t) [1]',fontsize=15,fontweight='bold')
            hax3.set_title('DC-ALICE',fontsize=20,fontweight='bold')
            hax4.set_xlim(0.1,51.), hax1.set_xticks(np.array([0.1,1.0,10.,50.]))
            hax4.set_ylim(10.**-6,10.**0), hax1.set_yticks(10.**np.arange(-6,1))
            hax4.set_xlabel('f [Hz]',fontsize=15,fontweight='bold')
            hax4.set_ylabel('A(f) [1]',fontsize=15,fontweight='bold')
            plt.savefig(os.path.join(outf,"fts_xw_%s_%u_%u.png"%(pfx,b,c)),\
                    bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"fts_xw_%s_%u_%u.eps"%(pfx,b,c)),\
                    format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
        zd_fl = zd_tf.reshape(-1,zd_tf.shape[2])
        zg_fl = zg_tf.reshape(-1,zd_tf.shape[2])
        #
        zi = ['z'+str(i) for i in range(zd_fl.shape[1])]
        zd_df = pd.DataFrame(zd_fl,columns=zi)
        zg_df = pd.DataFrame(zg_fl,columns=zi)

        Czz_zd = EmpiricalCovariance().fit(zd_fl)
        Czz_zg = EmpiricalCovariance().fit(zg_fl)

        z_correlation_matrix(zd_df,os.path.join(outf,"zd_{}".format(pfx)),\
                                                cov=Czz_zd.covariance_,prc=Czz_zd.precision_)
        z_correlation_matrix(zg_df,os.path.join(outf,"zg_{}".format(pfx)),\
                                                cov=Czz_zg.covariance_,prc=Czz_zg.precision_)
        for i,c in zip(range(len(zg_df.columns)),zg_df):
            plt.figure(figsize=(8,6))
            sns.distplot(zg_df[c],color=clr[i],label=r'$F_x(x)_{{ {{{:>d}}} }}$'.format(i),hist_kws={'alpha':.7},kde_kws={'linewidth':3})
            plt.title(r'Density Plot $F_x(x)_{{ {{{:>d}}} }}$'.format(i),fontsize=25)
            plt.xlim(-10.0,10.0)
            plt.ylim(0.0,1.0)
            plt.legend(fontsize=20) 
            plt.savefig(os.path.join(outf,"zg_hst_%s_z%u.png"%(pfx,i)),\
                    bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"zg_hts_%s_z%u.eps"%(pfx,i)),\
                    format='eps',bbox_inches='tight',dpi = 500)
            plt.close()
        for i,c in zip(range(len(zd_df.columns)),zd_df):
            plt.figure(figsize=(8,6))
            sns.distplot(zd_df[c],color=clr[i],label=r'$z_{{ {{{:>d}}} }}$'.format(i),hist_kws={'alpha':.7},kde_kws={'linewidth':3})
            plt.xlim(-10.0,10.0)
            plt.ylim(0.0,1.0)
            plt.title(r'Density Plot $z_{{ {{{:>d}}} }}$'.format(i),fontsize=25)
            plt.legend(fontsize=20) 
            plt.savefig(os.path.join(outf,"zd_hst_%s_z%u.png"%(pfx,i)),\
                    bbox_inches='tight',dpi = 500)
            plt.savefig(os.path.join(outf,"zd_hts_%s_z%u.eps"%(pfx,i)),\
                    format='eps',bbox_inches='tight',dpi = 500)

def z_correlation_matrix(df,figname,cov=None,prc=None):
    #pandas
    print("correlation_matrix ...")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('RdBu_r',200)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    plt.xticks(())
    plt.yticks(())
    plt.title('Feature Correlation Matrix',fontsize=25)
    labels=df.index.tolist()
    #ax1.set_xticklabels(labels,fontsize=20)
    #ax1.set_yticklabels(labels,fontsize=20)
    cbar=fig.colorbar(cax,ticks=np.linspace(-1.,1.,10).tolist())
    cbar.ax.tick_params(labelsize=15)
    plt.savefig(figname+"_cor.png",format='png',bbox_inches='tight',dpi = 500)
    #plt.savefig(figname+".eps",format='eps',bbox_inches='tight',dpi = 500)
    plt.close()

    if cov is not None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        vmax = cov.max()
        cmap = cm.get_cmap('RdBu_r',200)
        cax = ax1.imshow(cov,interpolation="nearest",vmin=-vmax,vmax=vmax,cmap=cmap)
        plt.xticks(())
        plt.yticks(())
        plt.title('Feature Covariance Matrix',fontsize=25)
        cbar=fig.colorbar(cax,ticks=np.linspace(-vmax,vmax,5).tolist())
        cbar.ax.tick_params(labelsize=15)
        plt.savefig(figname+"_cov.png",format='png',bbox_inches='tight',dpi = 500)
        #plt.savefig(figname+".eps",format='eps',bbox_inches='tight',dpi = 500)
        plt.close()
    if prc is not None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        vmax = 0.9*prc.max()
        cmap = cm.get_cmap('RdBu_r',200)
        cax = ax1.imshow(np.ma.masked_equal(prc, 0),interpolation="nearest",vmin=-vmax,vmax=vmax,cmap=cmap)
        plt.xticks(())
        plt.yticks(())
        plt.title('Feature Precision Matrix',fontsize=25)
        cbar=fig.colorbar(cax,ticks=np.linspace(-vmax,vmax,5).tolist())
        cbar.ax.tick_params(labelsize=15)
        if hasattr(ax1,'set_facecolor'):
            ax1.set_facecolor('.7')
        else:
            ax1.set_axis_bgcolor('.7')
        plt.savefig(figname+"_prc.png",format='png',bbox_inches='tight',dpi = 500)
        #plt.savefig(figname+".eps",format='eps',bbox_inches='tight',dpi = 500)
        plt.close()

def z_histogram(zt,zm,figname):
    print("plot histogram ...")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    zt = zt.stack().reset_index()
    zm = zm.stack().reset_index()
    cf = colors.to_rgba('orange')
    cf = list(cf)
    cf[-1]=0.5
    cf= tuple(cf)
    cb = colors.to_rgba('black')
    zt.hist(ax=ax1,bins=100,xlabelsize=12,ylabelsize=12,grid=False,label=r'$\mathbf{z}\sim\mathcal{N}(0,1)$',color=cf,density=True)
    zm.hist(ax=ax1,bins=100,xlabelsize=12,ylabelsize=12,grid=False,label=r'$F(\mathbf{x})$',color=cb,density=True)
    zi = np.linspace(-10.0,10.0,2001) 
    ax1.plot(zi,np.exp(-zi**2/2.)/np.sqrt(2.*np.pi),lw=4,color='gold')
    #import pdb
    #pdb.set_trace()
    print("[!]Mean of Variable zm ", zm.mean().values)
    ax1.axvline(x=zm.mean()[0],lw=4,color='cornflowerblue')
    ax1.axvline(x=zm.mean()[0]+zm.std().values[0],lw=2,color='cornflowerblue',ls='--')
    ax1.axvline(x=zm.mean()[0]-zm.std().values[0],lw=2,color='cornflowerblue',ls='--')
    ax1.text(-4.,1.2, r'$\mu={{ {{{:.1f}}} }};\sigma={{ {{{:.1f}}} }}$'.format(zm.mean().values.tolist()[0],zm.std().values.tolist()[0]),fontsize=20)
    plt.ylabel(r'$p(z_i)$',fontsize=20)
    plt.xlabel(r'$z_i$',fontsize=20)
    plt.ylim(-0.0,1.5)
    plt.xlim(-5.1,5.1)
    ax1.set_xticks(list(np.linspace(-5.0,5.0,11)))
    plt.title(r'A-posteriori distribution',fontsize=22)
    plt.legend(fontsize=20)
    plt.savefig(figname+".png",format='png',bbox_inches='tight',dpi = 500)
    #plt.savefig(figname+".eps",format='eps',bbox_inches='tight',dpi = 500)
    plt.close()

def discriminate_broadband_xz(DsXd,Dszd,Ddxz,Xd,Xdr,zd,zdr):
    
    # Discriminate real
    zrc = zcat(DsXd(Xd),Dszd(zdr))
    #DXz = Ddxz(zrc)
    DXz = torch.utils.checkpoint.checkpoint_sequential(functions=Ddxz, segments=8 , input=zrc)

    # Discriminate fake
    zrc = zcat(DsXd(Xdr),Dszd(zd))
    #DzX = Ddxz(zrc)
    DzX = torch.utils.checkpoint.checkpoint_sequential(functions=DsXd, segments=8 , input=zrc)

    return DXz,DzX

def discriminate_hybrid_xd(DsrXd,Xf,Xfr):
    Dreal = DsrXd(zcat(Xf,Xf ))
    Dfake = DsrXd(zcat(Xf,Xfr))
    return Dreal,Dfake

def seismo_test(tag,Fef,Gdd,Ghz,Fed,Ddxz,DsXd,Dszd,DsrXd,dev,trn_set,pfx,outf):
    sns.set(style="whitegrid")
    Fef.to(dev),Gdd.to(dev),Ghz.to(dev),Fed.to(dev)
    DsXd.to(dev),Dszd.to(dev),Ddxz.to(dev),DsrXd.to(dev)
    Fef.eval(),Gdd.eval(),Ghz.eval(),Fed.eval()
    DsXd.eval(),Dszd.eval(),Ddxz.eval(),DsrXd.eval()
    bsz = trn_set.batch_size
    idx = trn_set.dataset.indices.numpy()
    cht = trn_set.dataset.dataset.inpZ.shape[1]
    chf = trn_set.dataset.dataset.inpW.shape[1]
    if 'hybrid' in tag:
        Dtest=np.empty((len(idx),4))

        for b,batch in enumerate(trn_set):
            # Load batch
            xd_data,xf_data,zd_data,zf_data,_,_,_,*other = batch
            Xd = Variable(xd_data).to(dev) # BB-signal
            Xf = Variable(xf_data).to(dev) # LF-signal
            zd = Variable(zd_data).to(dev)
            zf = Variable(zf_data).to(dev)
            wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xf,wnxf)
            zfr = Fef(X_inp)
            zdr = Ghz(zcat(zfr,wnzf))
            z_inp = zcat(zdr,wnzd)
            Xr = Gdd(z_inp)
            Dxz,Dzx = discriminate_broadband_xz(DsXd,Dszd,Ddxz,Xd,Xr,zd,zdr)
            Dtest[b*bsz:(b+1)*bsz,0] = (Dxz.squeeze()+Dzx.squeeze()).mean(axis=-1).cpu().data.numpy()
            DXdr,DXdf = discriminate_hybrid_xd(DsrXd,Xd,Xr)
            Dtest[b*bsz:(b+1)*bsz,1] = DXdr.squeeze().mean(axis=-1).cpu().data.numpy()
            Dtest[b*bsz:(b+1)*bsz,2] = DXdf.squeeze().mean(axis=-1).cpu().data.numpy()
            DXdr,DXdf = discriminate_hybrid_xd(DsrXd,Xd,Xf)
            Dtest[b*bsz:(b+1)*bsz,3] = DXdf.squeeze().mean(axis=-1).cpu().data.numpy()

        print("size check: {}".format(Dtest.shape))
        Dt = pd.DataFrame(Dtest[:,0],columns=["Dxz+Dzx"])
        sns.kdeplot(Dt.loc[:,"Dxz+Dzx"],shade=True,color="dodgerblue",label=r"$D_{xz}(y,G_z(F_x(x)))+D_{xz}(G_y(G_z(F_x(x)),z^,))$")
        plt.title('Seismological Test', fontsize=25)
        plt.xticks(np.arange(0.,1.1,0.1))
        plt.legend()
        #Dtest['kind']=r"$G_y(F_F_y(y))$"
        #multivariateGrid('EG','PG','kind',df=egpg_df)
        plt.savefig(os.path.join(outf,"Dtest_%s.png"%(pfx['hybrid'])),\
                    bbox_inches='tight',dpi = 500)
        plt.savefig(os.path.join(outf,"Dtest_%s.eps"%(pfx['hybrid'])),\
                    format='eps',bbox_inches='tight',dpi = 500)
        plt.close()

        Dt = pd.DataFrame(Dtest[:,1:],columns=["Dreal","Dfake","Dfilt"])
        sns.kdeplot(Dt.loc[:,"Dreal"],shade=True,color="dodgerblue",label=r"$D_{y}^h(y)$")
        sns.kdeplot(Dt.loc[:,"Dfake"],shade=True,color="orange",label=r"$D_{y}^h(G_y(G_z(F_x(x))))$")
        sns.kdeplot(Dt.loc[:,"Dfilt"],shade=True,color="deeppink",label=r"$D_{y}^h(x)$")
        plt.title('Seismological Test', fontsize=25)
        #plt.xticks(np.arange(0.,1.1,0.1))
        plt.legend()
        plt.savefig(os.path.join(outf,"Dtest_hyb_%s.png"%(pfx['hybrid'])),\
                    bbox_inches='tight',dpi = 500)
        plt.savefig(os.path.join(outf,"Dtest_hyb_%s.eps"%(pfx['hybrid'])),\
                    format='eps',bbox_inches='tight',dpi = 500)
    if 'ann2bb' in tag:
        Dtest=np.empty((len(idx),6))
        sigma = torch.nn.Sigmoid()
        for b,batch in enumerate(trn_set):
            # Load batch
            xd_data,xf_data,zd_data,zf_data,_,_,_,xp_data,xs_data,xm_data = batch
            Xd = Variable(xd_data).to(dev)
            Xf = Variable(xf_data).to(dev)
            Xp = Variable(xp_data).to(dev)
            Xs = Variable(xs_data).to(dev)
            Xm = Variable(xm_data).to(dev)
            zd = Variable(zd_data).to(dev)
            zf = Variable(zf_data).to(dev)
            wnxd,wnzd,_ = noise_generator(Xd.shape,zd.shape,dev,app.RNDM_ARGS)
            wnxf,wnzf,_ = noise_generator(Xf.shape,zf.shape,dev,app.RNDM_ARGS)
            X_inp = zcat(Xf,wnxf)
            zfr = Fef(X_inp)
            zdr = Ghz(zcat(zfr,wnzf))
            zmr = Fed(zcat(Xm,wnxf))
            z_inp = zcat(zdr,wnzd)
            Xr = Gdd(z_inp)
            Dxz,Dzx = discriminate_broadband_xz(DsXd,Dszd,Ddxz,Xd,Xr,zd,zdr)
            Dtest[b*bsz:(b+1)*bsz,0] = (Dxz.squeeze()+Dzx.squeeze()).mean(axis=-1).cpu().data.numpy()
            Dxz,Dzx = discriminate_broadband_xz(DsXd,Dszd,Ddxz,Xd,Xm,zd,zmr)
            Dtest[b*bsz:(b+1)*bsz,1] = (Dxz.squeeze()+Dzx.squeeze()).mean(axis=-1).cpu().data.numpy()

            DXdr,DXdf = discriminate_hybrid_xd(DsrXd,Xd,Xr)
            Dtest[b*bsz:(b+1)*bsz,2] = sigma(DXdr).squeeze().mean(axis=-1).cpu().data.numpy()
            Dtest[b*bsz:(b+1)*bsz,3] = sigma(DXdf).squeeze().mean(axis=-1).cpu().data.numpy()

            DXdr,DXdf = discriminate_hybrid_xd(DsrXd,Xd,Xm)
            Dtest[b*bsz:(b+1)*bsz,4] = sigma(DXdf).squeeze().mean(axis=-1).cpu().data.numpy()

            DXdr,DXdf = discriminate_hybrid_xd(DsrXd,Xd,Xf)
            Dtest[b*bsz:(b+1)*bsz,5] = sigma(DXdf).squeeze().mean(axis=-1).cpu().data.numpy()
            
        Dt = pd.DataFrame(Dtest[:,0],columns=["Dxz+Dzx"])
        sns.kdeplot(Dt.loc[:,"Dxz+Dzx"],shade=True,color="dodgerblue",label=r"$D_{xz}(y,G_z(F_x(x)))+D_{xz}(G_y(G_z(F_x(x)),z^,))$")
        Dt = pd.DataFrame(Dtest[:,1],columns=["Dxz+Dzx"])
        sns.kdeplot(Dt.loc[:,"Dxz+Dzx"],shade=True,color="orange",label=r"$ANN2BB$")
        plt.title('Seismological Test', fontsize=25)
        plt.xticks(np.arange(0.9,1.1,0.1))
        plt.legend()
        #Dtest['kind']=r"$G_y(F_F_y(y))$"
        #multivariateGrid('EG','PG','kind',df=egpg_df)
        plt.savefig(os.path.join(outf,"Dtest_%s.png"%(pfx['ann2bb'])),\
                    bbox_inches='tight',dpi = 500)
        plt.savefig(os.path.join(outf,"Dtest_%s.eps"%(pfx['ann2bb'])),\
                    format='eps',bbox_inches='tight',dpi = 500)
        plt.close()

        Dt = pd.DataFrame(Dtest[:,2:],columns=["Dreal","Dfake","ANN2BB","Dfilt"])
        #    r"$D_{xz}(y,z^,)+D_{xz}(y,z^,)$"])
        #sns.kdeplot(Dt.loc[:,"Dreal"],shade=True,color="dodgerblue",label=r"$D_{y}^h(y)$")
        #sns.kdeplot(Dt.loc[:,"Dfake"],shade=True,color="orange",label=r"$D_{y}^h(G_y(G_z(F_x(x))))$")
        #sns.kdeplot(Dt.loc[:,"ANN2BB"],shade=True,color="springgreen",label=r"$D_{y}^h(ANN2BB)$")
        #hax=sns.kdeplot(Dt.loc[:,"Dfilt"],shade=True,color="deeppink",label=r"$D_{y}^h(x)$")
        sns.distplot(Dt.loc[:,"Dreal"],color="dodgerblue",label=r"$D_{y}^h(y)$")
        sns.distplot(Dt.loc[:,"Dfake"],color="orange",label=r"$D_{y}^h(G_y(G_z(F_x(x))))$")
        sns.distplot(Dt.loc[:,"ANN2BB"],color="springgreen",label=r"$D_{y}^h(ANN2BB)$")
        hax=sns.distplot(Dt.loc[:,"Dfilt"],color="deeppink",label=r"$D_{y}^h(x)$")
        #hax.set_xscale('log')
        plt.title('Seismological Test', fontsize=25)
        #plt.xticks(np.arange(0.,1.1,0.1))
        plt.legend()
        plt.savefig(os.path.join(outf,"Dtest_hyb_%s.png"%(pfx['ann2bb'])),\
                    bbox_inches='tight',dpi = 500)
        plt.savefig(os.path.join(outf,"Dtest_hyb_%s.eps"%(pfx['ann2bb'])),\
                    format='eps',bbox_inches='tight',dpi = 500)
        print("plot Gy(Fy(y)) gofs")


def plot_eg_pg(x,y, outf,pfx=''):
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=(8, 8))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    # the scatter plot:
    ax_scatter.scatter(x, y)
    

    # now determine nice limits by hand:
    binwidth = 0.25
    # lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    lim = 10
    ax_scatter.set_xlim((0, lim))
    ax_scatter.set_ylim((0, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    plt.savefig(os.path.join(outf,"gof_eg_pg%s.png"%(pfx)),\
                        bbox_inches='tight',dpi = 300)
    print("saving gof_eg_pg%s.png"%(pfx))
    plt.close()
