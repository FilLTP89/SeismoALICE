import os
import pdb
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from obspy.signal.tf_misfit import plot_tf_gofs, eg,pg
from scipy.fft import fft, fftfreq, fftshift
from tools.generate_noise import noise_generator
from common.common_nn import zcat
from common.common_torch import tfft
from tqdm import  tqdm,trange


def plot_signal_and_reconstruction(vld_set,encoder,
        decoder, outf, opt, device='cuda',pfx='investigate'):
    # extract data : 
    # breakpoint()
    encoder.eval()
    decoder.eval()
    sns.set(style="whitegrid")
    clr = sns.color_palette('tab10',5)
    cnt = 0
    t = np.linspace(0,40.96,4096) 
    vtm = torch.load(os.path.join(opt.dataroot,'vtm.pth'))  
    freq = fftshift(fftfreq(t.shape[-1]))  
    rndm_args = {'mean': 0, 'std': 1.0}
    bar = tqdm(vld_set)
    for b, batch in enumerate(bar):
        
        xd_data,zd_data, *others = batch
        Xt = xd_data.to(device)
        zt = zd_data.to(device)

        #noise
        wnx,*others = noise_generator(Xt.shape,zt.shape,device,rndm_args)
        X_inp = zcat(Xt,wnx.to(device))
        #encoding
        zy, zyx = encoder(X_inp)
        z_inp = zcat(zyx,zy)

        #decoding
        Xr = decoder(z_inp)

        Xt_fsa = tfft(Xt,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        Xr_fsa = tfft(Xr,vtm[1]-vtm[0]).cpu().data.numpy().copy()
        vfr    = np.arange(0,vtm.size,1)/(vtm[1]-vtm[0])/(vtm.size-1)

        Xt = Xt.cpu().data.numpy().copy()
        Xr = Xr.cpu().data.numpy().copy()

        plt.figure(b)

        for (io, ig) in zip(range(Xt.shape[0]),range(Xr.shape[0])):
            ot, gt = Xt[io,1,:], Xr[io,1,:]
            of, gf = Xt_fsa[io,1,:], Xr_fsa[ig,1,:]

            cnt+=1
            
            hgof = plot_tf_gofs(ot,gt,dt=vtm[1]-vtm[0],t0=0.0,fmin=0.1,fmax=30.0,
                    nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.,left=0.1,
                    bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2, w_1=0.2,w_2=0.6,
                    w_cb=0.01, d_cb=0.0,show=False,plot_args=['k', 'r', 'b'],
                    ylim=0., clim=0.)
            
            plt.savefig(os.path.join(outf,"gof_bb_aae_%s_%u_%u.png"%(pfx,cnt,io)),\
                            bbox_inches='tight',dpi = 300)

            fig,(hax0,hax1) = plt.subplots(2,1,figsize=(6,8))
                # hax0.plot(vtm,gt,color=clr[3],label=r'$G_t(zcat(F_x(\mathbf{y},N(\mathbf{0},\mathbf{I})))$',linewidth=1.2)
            hax0.plot(vtm,ot,color=clr[0],label=r'$\mathbf{x}$',linewidth=1.2, alpha=0.70)
            hax0.plot(vtm,gt,color=clr[3],label=r'$G_y(F(\mathbf{x})$',linewidth=1.2)
            hax1.loglog(vfr,of,color=clr[0],label=r'$\mathbf{x}$',linewidth=2)
            # hax1.loglog(vfr,ff,color=clr[1],label=r'$\mathbf{x}$',linewidth=2)
            hax1.loglog(vfr,gf,color=clr[3],label=r'$G_y(F_t(\mathbf{x}))$',linewidth=2)
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


def model_visualization_encoder_unic(trn_set, model, opt, outf):
    model.eval()
    i, j = 0, 0
    vtm = torch.load(os.path.join(opt.dataroot,'vtm.pth'))
    batch = next(iter(trn_set))

    x,*others =  batch
    signals   = x.cuda()

    pdb.set_trace()
    seq = next(iter(model.children()))
    # master = next(iter(seq.children()))
    # master = master.cuda()
    # x_temp = master(x)

    plot_signal_layer(layer = seq.master,pfx='master')
    # cnt = 1
    # hfig, (p0, p1) = plt.subplots(2,1, figsize=(6,8))
    # for layer in seq.master.children():
    #     signals = layer(signals)
    #     cnt = plot(x = signals, layer = layer, opt=opt,vtm = vtm, 
    #         layer_val=cnt, pfx = 'master', p0=p0, p1 = p1)
    # hfig.savefig(os.path.join(opt.outf,"cnn_branch_%s_layer_%u.png"%('master',cnt)),\
    #                             bbox_inches='tight',dpi = 500)
    # plt.close()

    zyx = signals
    cnt = 1
    hfig, (p0, p1) = plt.subplots(2,1, figsize=(6,8))
    for layer in seq.cnn_common.children():
        zyx = layer(zyx)
        cnt = plot(x = zyx, layer = layer, opt=opt,vtm = vtm, 
            layer_val=cnt, pfx = 'cnn_common', p0=p0, p1 = p1)
    hfig.savefig(os.path.join(opt.outf,"cnn_branch_%s_layer_%u.png"%('cnn_common',cnt)),\
                                bbox_inches='tight',dpi = 500)
    plt.close()

    zy = signals
    cnt = 1
    hfig, (p0, p1) = plt.subplots(2,1, figsize=(6,8))
    for layer in seq.cnn_broadband.children():
        zy = layer(zy)
        cnt = plot(x = zy, layer = layer, opt=opt,vtm = vtm, 
            layer_val=cnt, pfx = 'cnn_broadband', p0=p0, p1 = p1)
    hfig.savefig(os.path.join(opt.outf,"cnn_branch_%s_layer_%u.png"%('cnn_broadband',cnt)),\
                                bbox_inches='tight',dpi = 500)
    plt.close()


def plot_signal_layer(layer,pfx):
    cnt = 1
    hfig, (p0, p1) = plt.subplots(2,1, figsize=(6,8))
    for layer in seq.master.children():
        signals = layer(signals)
        cnt = plot(x = signals, layer = layer, opt=opt,vtm = vtm, 
            layer_val=cnt, pfx = pfx, p0=p0, p1 = p1)
    hfig.savefig(os.path.join(opt.outf,"cnn_branch_%s_layer_%u.png"%(pfx,cnt)),\
                                bbox_inches='tight',dpi = 500)
    plt.close()

def plot(x,layer,opt,vtm, layer_val, pfx, p1, p0):
    classname = layer.__class__.__name__
    if classname == 'Conv1d' or classname == 'ConvTranspose1d':
        
        t   = np.linspace(vtm[0],vtm[-1],x.shape[-1])
        vfr = np.arange(0,t.size,1)/(t[1]-t[0])/(t.size-1)
        xf  = tfft(x,t[1]-t[0]).cpu().data.numpy().copy()
        st = x[1,1,: ].cpu().data.numpy().copy()
        sf = xf[1,1,:]
        p0.plot(t,st, label='signal_layer %u'%(layer_val))
        p1.loglog(vfr,sf,label='signal_layer %u'%(layer_val))
       
        print("cnn_branch_%s_layer_%u.png"%(pfx,layer_val))
        layer_val = layer_val +1
    return layer_val

def visualize_gradients(net,loss, color="C0"):
    """
    Args:
        net: Object of class BaseNetwork
        color: Color in which we want to visualize the histogram (for easier separation of activation functions)
    """
    net.eval()
    act_fn_by_name = {"sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU, "leakyrelu": LeakyReLU, "elu": ELU, "swish": Swish}
 
    loss.backward()
    # We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots
    grads = {
        name: params.grad.data.view(-1).cpu().clone().numpy()
        for name, params in net.named_parameters()
        if "weight" in name
    }
    net.zero_grad()

    # Plotting
    columns = len(grads)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3.5, 2.5))
    fig_index = 0
    for key in grads:
        key_ax = ax[fig_index % columns]
        sns.histplot(data=grads[key], bins=30, ax=key_ax, color=color, kde=True)
        key_ax.set_title(str(key))
        key_ax.set_xlabel("Grad magnitude")
        fig_index += 1
    fig.suptitle(
        f"Gradient magnitude distribution for activation function {net.config['act_fn']['name']}", fontsize=14, y=1.05
    )
    fig.subplots_adjust(wspace=0.45)
    plt.show()
    plt.savefig("./imgs/gradient.png")
    plt.close()




