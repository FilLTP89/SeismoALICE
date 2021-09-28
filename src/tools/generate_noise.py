# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Main file to generate noise with predefined SNR
"""
u'''Required modules'''
import numpy as np
from scipy.signal import filtfilt,butter
import torch
from torch import FloatTensor as tFT
from common.common_nn import  Variable
import math

u'''General informations'''
__author__ = "Filippo Gatti & Didier Clouteau"
__copyright__ = "Copyright 2018, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"

def SNR(ths,noise,dtm):
    Signal = np.sum(np.abs(np.fft.fft(ths)*dtm)**2)/len(np.fft.fft(ths))
    Noise = np.sum(np.abs(np.fft.fft(noise)*dtm)**2)/len(np.fft.fft(noise))
    return (10 * np.log10(Signal/Noise))


def Gauss_noise(pc,ntm,scl):
    wnz = np.random.normal(scale=pc*scl, size=ntm)
    return wnz

def scale_Gauss_noise(wnz,scl):
    return wnz*scl

def filter_Gauss_noise(wnz=np.empty((1,1)),ffr=0.16):
    wnz = wnz * cosine_taper(wnz.size, p=0.01)
    wnz = filter_signal(wnz,ffr,typ='highpass')
    wnz = wnz * cosine_taper(wnz.size, p=0.01)
#     z, p, k = iirfilter(4,ffr,btype='highpass',\
#                         ftype='butter',output='zpk')
#     sos = zpk2sos(z, p, k)
#     wnz = sosfilt(sos, wnz)
#     wnz = sosfilt(sos, wnz[::-1])[::-1]
    return wnz

def filter_signal(ths,ffr,typ='lowpass'):
    pad = int(1.5*2*2/ffr)
    ths = np.pad(ths,pad,'constant')
    b,a = butter(4,ffr,typ)
    ths = filtfilt(b,a,ths)
    ths = ths[pad:-pad]
    return ths
    

def noisy_sig(dtm=0.008,ths=np.empty((1,1)),ffr=0.16,\
              pc=2e-1,wnz=None):
    scl = np.max(ths)
    if wnz is None:
        ntm = len(ths)
        wnz = Gauss_noise(pc,ntm,scl)
        wnz = filter_Gauss_noise(wnz,ffr)
    else:
        wnz = scale_Gauss_noise(wnz,scl)
        
    ths = filter_signal(ths,ffr)
    wnz_fft = np.fft.fft(wnz)*dtm
    ths_fft = np.fft.fft(ths)*dtm
    
#     frf = ffr/2./dtm
#     dfr = 1./dtm/ths.size
#     nfa = int(0.9*frf/dfr)
#     nfb = int(1.1*frf/dfr)
#     fac = np.pi/dfr/(nfb-nfa-1)
#     WLF = np.zeros((ntm,))
#     WLF[:nfa] = 1.0
#     WLF[nfa:nfb] = 0.5*(1.0+np.cos(fac*dfr*np.arange(0,nfb-nfa)))
#     WHF = 1.0-WLF
#     ths_fft *= WLF
#     wnz_fft *= WHF

    from matplotlib import pyplot as plt
    plt.plot(np.abs(wnz_fft),'b')
    plt.plot(np.abs(ths_fft),'r')
    plt.plot(np.abs(ths_fft+wnz_fft),'k')
    try:
        ths = np.fft.ifft(ths_fft+wnz_fft)/dtm
    except:
        print("OK")
    #plt.plot(np.abs(np.fft.fft(ths)*dtm),'g')
    return wnz,ths

def pollute_ths(dtm=0.008,ths=np.empty((1,1)),ffr=0.16,\
                pc=2e-1,wnz=None):
    # HIGHPASS FILTERED GAUSSIAN NOISE
    wnz,ths = noisy_sig(dtm=dtm,ths=ths,ffr=ffr,\
                        pc=pc,wnz=wnz)
    return wnz,ths

def pollute_ths_tensor(dtm=0.008,ths=np.empty((1,1)),ffr=0.16,\
                       pc=2e-1,wnz=None):
    ths_wnz = ths.cpu().data.numpy().copy()
        
    for i in range(ths_wnz.shape[0]):
        for j in range(ths_wnz.shape[1]):
            _,y = pollute_ths(dtm=dtm,ths=ths_wnz[i,j,:],\
                              ffr=ffr,pc=pc,wnz=wnz)
            ths_wnz[i,j,:] = y.reshape(ths_wnz.shape[2])
            
    ths_wnz = torch.from_numpy(ths_wnz)
    return ths_wnz

def filter_ths_tensor(dtm=0.008,ths=np.empty((1,1)),ffr=0.16,\
                       pc=2e-1,wnz=None):
    ths_wnz = ths.cpu().data.numpy().copy()
        
    for i in range(ths_wnz.shape[0]):
        for j in range(ths_wnz.shape[1]):
            y = filter_signal(ths=ths_wnz[i,j,:],ffr=ffr)
            ths_wnz[i,j,:] = y.reshape(ths_wnz.shape[2])
            
    #wnz_wnz = torch.from_numpy(wnz_wnz)
    ths_wnz = torch.from_numpy(ths_wnz)
    
    return ths_wnz


def cosine_taper(npts, p=0.1, freqs=None, flimit=None, halfcosine=True,
                 sactaper=False):
    """
    Cosine Taper.

    :type npts: int
    :param npts: Number of points of cosine taper.
    :type p: float
    :param p: Decimal percentage of cosine taper (ranging from 0 to 1). Default
        is 0.1 (10%) which tapers 5% from the beginning and 5% form the end.
    :rtype: float NumPy :class:`~numpy.ndarray`
    :return: Cosine taper array/vector of length npts.
    :type freqs: NumPy :class:`~numpy.ndarray`
    :param freqs: Frequencies as, for example, returned by fftfreq
    :type flimit: list or tuple of floats
    :param flimit: The list or tuple defines the four corner frequencies
        (f1, f2, f3, f4) of the cosine taper which is one between f2 and f3 and
        tapers to zero for f1 < f < f2 and f3 < f < f4.
    :type halfcosine: bool
    :param halfcosine: If True the taper is a half cosine function. If False it
        is a quarter cosine function.
    :type sactaper: bool
    :param sactaper: If set to True the cosine taper already tapers at the
        corner frequency (SAC behavior). By default, the taper has a value
        of 1.0 at the corner frequencies.

    .. rubric:: Example

    >>> tap = cosine_taper(100, 1.0)
    >>> tap2 = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 50)))
    >>> np.allclose(tap[0:50], tap2)
    True
    >>> npts = 100
    >>> p = 0.1
    >>> tap3 = cosine_taper(npts, p)
    >>> (tap3[int(npts*p/2):int(npts*(1-p/2))]==np.ones(int(npts*(1-p)))).all()
    True
    """
    if p < 0 or p > 1:
        msg = "Decimal taper percentage must be between 0 and 1."
        raise ValueError(msg)
    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)

    if freqs is not None and flimit is not None:
        fl1, fl2, fl3, fl4 = flimit
        idx1 = np.argmin(abs(freqs - fl1))
        idx2 = np.argmin(abs(freqs - fl2))
        idx3 = np.argmin(abs(freqs - fl3))
        idx4 = np.argmin(abs(freqs - fl4))
    else:
        idx1 = 0
        idx2 = frac - 1
        idx3 = npts - frac
        idx4 = npts - 1
    if sactaper:
        # in SAC the second and third
        # index are already tapered
        idx2 += 1
        idx3 -= 1

    # Very small data lengths or small decimal taper percentages can result in
    # idx1 == idx2 and idx3 == idx4. This breaks the following calculations.
    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1

    # the taper at idx1 and idx4 equals zero and
    # at idx2 and idx3 equals one
    cos_win = np.zeros(npts)
    if halfcosine:
        # cos_win[idx1:idx2+1] =  0.5 * (1.0 + np.cos((np.pi * \
        #    (idx2 - np.arange(idx1, idx2+1)) / (idx2 - idx1))))
        cos_win[idx1:idx2 + 1] = 0.5 * (
            1.0 - np.cos((np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) /
                          (idx2 - idx1))))
        cos_win[idx2 + 1:idx3] = 1.0
        


        cos_win[idx3:idx4 + 1] = 0.5 * (
            1.0 + np.cos((np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) /
                          (idx4 - idx3))))
    else:
        cos_win[idx1:idx2 + 1] = np.cos(-(
            np.pi / 2.0 * (float(idx2) -
                           np.arange(idx1, idx2 + 1)) / (idx2 - idx1)))
        cos_win[idx2 + 1:idx3] = 1.0
        cos_win[idx3:idx4 + 1] = np.cos((
            np.pi / 2.0 * (float(idx3) -
                           np.arange(idx3, idx4 + 1)) / (idx4 - idx3)))

    # if indices are identical division by zero
    # causes NaN values in cos_win
    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0
    return cos_win


def latent_resampling(z_hat,nz,noise):
    mu, sigma = z_hat[:, :nz,:], z_hat[:,nz:,:].exp()
    z_hat = mu + sigma * noise.expand_as(sigma)
    return z_hat

def noise_generator(Xshape,zshape,device,rndm_args):
    wnz = tFT(*zshape).to(device)
    wnx = tFT(*Xshape).to(device)
    wn1 = tFT(Xshape[0],1,1).to(device)
    wn1 = Variable(wn1).to(device)
    wnz = Variable(wnz).to(device)
    wnx = Variable(wnx).to(device)
    # Generate Noise
    with torch.no_grad():
        wnz.resize_(*zshape).normal_(**rndm_args)
        wnx.resize_(*Xshape).normal_(**rndm_args)
        wn1.resize_(Xshape[0],1,1).normal_(**rndm_args)
    return wnx,wnz,wn1 

def lfilter(waveform, a_coeffs, b_coeffs):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    r"""Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`.  Must be normalized to -1 to 1.
        a_coeffs (torch.Tensor): denominator coefficients of difference equation of dimension of `(n_order + 1)`.
                                Lower delays coefficients are first, e.g. `[a0, a1, a2, ...]`.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (torch.Tensor): numerator coefficients of difference equation of dimension of `(n_order + 1)`.
                                 Lower delays coefficients are first, e.g. `[b0, b1, b2, ...]`.
                                 Must be same size as a_coeffs (pad with 0's as necessary).

    Returns:
        output_waveform (torch.Tensor): Dimension of `(..., time)`.  Output will be clipped to -1 to 1.

    """

    dim = waveform.dim()

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    assert(a_coeffs.size(0) == b_coeffs.size(0))
    assert(len(waveform.size()) == 2)
    assert(waveform.device == a_coeffs.device)
    assert(b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    assert(n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample + n_order - 1, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample + n_order - 1, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip order, repeat, and transpose
    a_coeffs_filled = a_coeffs.flip(0).repeat(n_channel, 1).t()
    b_coeffs_filled = b_coeffs.flip(0).repeat(n_channel, 1).t()

    # Set up a few other utilities
    a0_repeated = torch.ones(n_channel, dtype=dtype, device=device) * a_coeffs[0]
    ones = torch.ones(n_channel, n_sample, dtype=dtype, device=device)

    for i_sample in range(n_sample):

        o0 = torch.zeros(n_channel, dtype=dtype, device=device)

        windowed_input_signal = padded_waveform.clone()[:, i_sample:(i_sample + n_order)]
        windowed_output_signal = padded_output_waveform.clone()[:, i_sample:(i_sample + n_order)]

        o0.add_(torch.diag(torch.mm(windowed_input_signal.clone(), b_coeffs_filled)))
        o0.sub_(torch.diag(torch.mm(windowed_output_signal.clone(), a_coeffs_filled)))

        o0.div_(a0_repeated)

        padded_output_waveform[:, i_sample + n_order - 1] = o0.clone()

    output = torch.min(
        ones, torch.max(ones * -1, padded_output_waveform.clone()[:, (n_order - 1):])
    )

    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])

    return output

def biquad(waveform, b0, b1, b2, a0, a1, a2):
    # type: (Tensor, float, float, float, float, float, float) -> Tensor
    r"""Perform a biquad filter of input tensor.  Initial conditions set to 0.
    https://en.wikipedia.org/wiki/Digital_biquad_filter

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        b0 (float): numerator coefficient of current input, x[n]
        b1 (float): numerator coefficient of input one time step ago x[n-1]
        b2 (float): numerator coefficient of input two time steps ago x[n-2]
        a0 (float): denominator coefficient of current output y[n], typically 1
        a1 (float): denominator coefficient of current output y[n-1]
        a2 (float): denominator coefficient of current output y[n-2]

    Returns:
        output_waveform (torch.Tensor): Dimension of `(..., time)`
    """

    device = waveform.device
    dtype = waveform.dtype

    output_waveform = lfilter(
        waveform.clone(),
        torch.tensor([a0, a1, a2], dtype=dtype, device=device),
        torch.tensor([b0, b1, b2], dtype=dtype, device=device)
    )
    return output_waveform

def _dB2Linear(x):
    # type: (float) -> float
    return math.exp(x * math.log(10) / 20.0)

def lowpass_biquad(waveform, sample_rate, cutoff_freq, Q=0.707):
    # type: (Tensor, int, float, float) -> Tensor
    r"""Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float): https://en.wikipedia.org/wiki/Q_factor

    Returns:
        output_waveform (torch.Tensor): Dimension of `(..., time)`
    """
    GAIN = 1.
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)
