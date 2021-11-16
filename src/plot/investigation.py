import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from tools.generate_noise import noise_generator
from common.common_nn import zcat
from common.common_torch import tfft





def plot_signal_and_reconstruction(vld_set,encoder,decoder, outf,device='cuda'):
    # extract data : 
    # breakpoint()
    t = np.linspace(0,40.96,4096)   
    freq = fftshift(fftfreq(t.shape[-1]))  
    rndm_args = {'mean': 0, 'std': 1}

    for b, batch in enumerate(vld_set):
        xd_data, zd_data = batch
        Xt = xd_data.to(device)
        zt = zd_data.to(device)

        #noise
        wnx,wnz,wn1 = noise_generator(Xt.shape,zt.shape,device,rndm_args)
        X_inp = zcat(Xt,wnx.to(device))
        #encoding
        ztr   = encoder(X_inp)
        
        #noise
        z_inp = zcat(ztr,wnz)

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


def model_visualization(train_set, model):
    model.eval()
    hfig, (p0, p1) = plt.subplots(2,1, figsize=(6,8))
    for model_child in model.children(): 
        x = model_child(train_set)
        plot(model_child, x, p0, p1)
        for layer in model_child.children(): 
            x = layer(train_set)
            plot(layer,x, p0, p1) 



def plot(layer, x, p0, p1):
    classname = layer.__class__.__name__ 
    if layer.find('Conv1d') or layer.find('ConvTranspose1d'):
        #number of time steps
        Nt = x.shape[2]
        #time figure
        t  = np.linspace(0,40.06, Nt)
        st = x[1,1,:].cpu().data.numpy().copy()
        p0.plot(t,st, label = 'layer')

        #frequence figure
        freq = fftshift(fftfreq(t.shape[-1])) 
        sf = tfft(Xt,0.01).cpu().data.numpy().copy()
        p1.plot(freq,sf, label = 'layer')

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




