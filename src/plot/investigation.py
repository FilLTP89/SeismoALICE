import os
import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from tools.generate_noise import noise_generator
from common.common_nn import zcat
from common.common_torch import tfft


def plot_signal_and_reconstruction(vld_set,encoder,
        decoder, outf, opt, device='cuda'):
    # extract data : 
    # breakpoint()
    t = np.linspace(0,40.96,4096) 
    vtm = torch.load(os.path.join(opt.dataroot,'vtm.pth'))  
    freq = fftshift(fftfreq(t.shape[-1]))  
    rndm_args = {'mean': 0, 'std': 1.0}

    for b, batch in enumerate(vld_set):
        _, xd_data, *others = batch
        Xt = xd_data.to(device)
        zt = zd_data.to(device)

        #noise
        wnx,_ = noise_generator(Xt.shape,zt.shape,device,rndm_args)
        X_inp = zcat(Xt,wnx.to(device))
        #encoding
        zy, zyx = encoder(X_inp)
        z_inp = zcat(zyx,zy)

        #decoding
        Xr = decoder(z_inp)

        Xt_fsa = tfft(Xt,0.01).cpu().data.numpy().copy()
        Xr_fsa = tfft(Xr,0.01).cpu().data.numpy().copy()

        Xt = Xt.cpu().data.numpy().copy()
        Xr = Xr.cpu().data.numpy().copy()

        plt.figure(b)

        for (io, ig) in zip(range(Xt.shape[0]),range(Xr.shape[0])):
            ot, gt = Xt[io,1,:], Xr[io,1,:]
            of, gf = Xt_fsa[io,1,:], Xr_fsa[ig,1,:]

            hfig, (p0, p1) = plt.subplots(2,1, figsize=(6,8))
            
            p0.plot(t, ot, label=r'$\mathbf{x}$',linewidth=1.2)
            p0.plot(t, gt, label=r'$\mathbf{\hat{x}}$',linewidth=1.2)
            p0.set_xlabel('t')
            p0.set_ylabel('A(t)')

            p1.loglog(freq, of, label=r'$\mathbf{x}$',linewidth=1.2)
            p1.loglog(freq, gf, label=r'$\mathbf{\hat{x}}$',linewidth=1.2)
            p0.set_xlabel('f')
            p0.set_ylabel('A(f)')
            file = outf+"/signal_{0}_{1}_{2}.png".format(b,io,ig)
            print(file)
            plt.savefig(file)


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




