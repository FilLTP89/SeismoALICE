import numpy as np
import pandas as pd
import sklearn as skl
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.model_selection import train_test_split
import h5py

import matplotlib.pyplot as plt
from pyts.decomposition import SingularSpectrumAnalysis

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column
from bokeh.models import CustomJS, Slider, ColumnDataSource
import bokeh.io
from bokeh.io import curdoc
from bokeh.models.renderers import GlyphRenderer

import yaml
import os

from numpy.fft import fft
from numpy.fft import ifft

import time as timer_sec

later = 0.0


# TO EJECTUTE
# bokeh serve --show .\ssa_app_civa.py
def remote_jupyter_proxy_url(port):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.

    If port is None we're asking about the URL
    for the origin header.
    """
    base_url = os.environ['EXTERNAL_URL']
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ['JUPYTERHUB_SERVICE_PREFIX']
    proxy_url_path = 'proxy/%d' % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url


def set_legend_font(fig, font_size="12pt"):
    fig.legend.location = "top_right"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label_text_font_size = font_size
    fig.yaxis.axis_label_text_font_size = font_size
    fig.title.text_font_size = font_size
    fig.xaxis.major_label_text_font_size = font_size
    fig.yaxis.major_label_text_font_size = font_size


def tfft(ths, dtm, nfr=None):
    if nfr is None:
        fsa = (abs(fft(ths, axis=-1) * dtm))
    else:
        fsa = (abs(fft(ths, axis=-1, n=nfr) * dtm))
    return fsa

    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.

    If port is None we're asking about the URL
    for the origin header.
    """
    base_url = os.environ['EXTERNAL_URL']
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ['JUPYTERHUB_SERVICE_PREFIX']
    proxy_url_path = 'proxy/%d' % port

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url


# Reading CIVA
hf = h5py.File(r'ut_dataset/Hybrid_civa_oofe_1024_butter.hdf5', 'r')
tmpciva = hf['civa']['sample 1']['data']
print('CIVA dataset: ', tmpciva)

civa = tmpciva[()]
print('CIVA dim: ', civa.shape)
print('CIVA ex: ', civa[0, 0, :], len(civa[0, 0, :]))
# civa[ex,0-2,1024]

# Reading OOFE
print('OOFE dataset:', hf['oofe']['sample 10']['data'])

oofe = hf['oofe']['sample 1']['data']
for i in range(1, 10):
    tmpoofe = hf['oofe']['sample ' + str(i + 1)]['data']
    oofe = np.concatenate((oofe, tmpoofe[()]), axis=0)

print('OOFE dim: ', oofe.shape)
print('OOFE ex: ', oofe[0][0][:], len(oofe[0][0][:]))

timestep = hf['time']['values'][1] - hf['time']['values'][0]
print('time step: ', timestep)  # duration: 5e-6
hf.close()

fs = 1 / float(timestep)
N = 1024
time = np.arange(N) / float(fs)
freq = np.arange(0, 1024, 1) / timestep / (1024 - 1)

# Adding noise using target SNR
# Set a target SNR
target_snr_db = 30

# Calculate signal power and convert to dB 
sig_avg = np.mean(civa[:, 0, :])
sig_avg_db = 10 * np.log10(sig_avg)

i = 0
civa_noise = np.zeros(shape=civa[:, 0, :].shape)
for ex in civa[:, 0, :]:
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg), 1024)
    civa_noise[i, :] = ex + noise
    i = i + 1

# Calculate signal power and convert to dB 
target_snr_db = 30
sig_avg = np.mean(oofe[:, 0, :])
sig_avg_db = 10 * np.log10(sig_avg)

i = 0
oofe_noise = np.zeros(shape=oofe[:, 0, :].shape)
for ex in oofe[:, 0, :]:
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg - target_snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg), 1024)
    oofe_noise[i, :] = ex + noise
    i = i + 1



def remove_glyphs(figure, glyph_name_list):
    renderers = figure.select(dict(type=GlyphRenderer))
    for r in renderers:
        if r.name in glyph_name_list:
            col = r.glyph.y
            r.data_source.data[col] = [np.nan] * len(r.data_source.data[col])


def get_example_reconstruction(ex):
    # get SSA example civa_r data
    # print(ex)

    # Singular Spectrum Analysis
    ssa = SingularSpectrumAnalysis(window_size, groups=n_groups)
    # civa[ex,0,time]
    X_ssa = ssa.transform(civa_noise[ex].reshape(1, -1))  # single sample

    # Reconstruction of set
    global keep_groups  # int(n_groups*0.9) #n groups to draw (4 last try)
    del_groups = n_groups - keep_groups

    civa_r = np.zeros(len(civa_noise[0]))
    for i in range(len(groups) - del_groups):
        civa_r = np.add(civa_r, X_ssa[i])

    data_civa = (dict(freq=np.arange(0, 1024, 1) / (timestep) / (1024 - 1),
                      f1=tfft(civa_noise[ex, :], timestep),
                      f2=tfft(civa_r, timestep),
                      f3=tfft(civa[ex, 0, :], timestep),
                      civa_noise=civa_noise[ex],
                      civa_r=civa_r,
                      time=time))
    return data_civa, X_ssa


# We decompose the time series into three subseries
window_size = 64
n_groups = 64

if window_size % n_groups == 0:
    groups = [np.arange(i, i + int(window_size / n_groups)) for i in range(0, window_size, int(window_size / n_groups))]
else:
    print("Change windows-group relation")

# Singular Spectrum Analysis
ssa = SingularSpectrumAnalysis(window_size, groups=n_groups)
# civa[ex,0,time]
# ssa.fit_transform(civa_train)
X_ssa = ssa.transform(civa_noise[0].reshape(1, -1))  # single sample

# Reconstruction of set
keep_groups = 4  # int(n_groups*0.9) #n groups to draw (4 last try)
del_groups = n_groups - keep_groups
ex = 10  # example

civa_r = np.zeros(len(civa_noise[0]))
for i in range(len(groups) - del_groups):
    civa_r = np.add(civa_r, X_ssa[i])

# PLOT useful groups

color = ["orange", "red", "blue", "green", "yellow"]

p = figure(title="CIVA_SSA", x_axis_label='t[s]', y_axis_label='A[a.u.]', plot_width=800, plot_height=400,x_range=(0,5e-6))
p_r = figure(title="CIVA_reconstructed_test", x_axis_label='t[s]', y_axis_label='A[a.u.]', plot_width=800,
             plot_height=400,x_range=(0,5e-6))
p_n = figure(title="CIVA_SSA_del_group", x_axis_label='t[s]', y_axis_label='A[a.u.]', plot_width=800, plot_height=400,x_range=(0,5e-6))
p_f = figure(title="CIVA_fft", x_axis_label='f[Hz]', y_axis_label='A[a.u.]', plot_width=800, plot_height=400,
             y_axis_type="log", x_axis_type="log", x_range=(1e5, 1e8))

source_civa = ColumnDataSource(dict(freq=np.arange(0, 1024, 1) / (timestep) / (1024 - 1),
                                    f1=tfft(civa_noise[0, :], timestep),
                                    f2=tfft(civa_r, timestep),
                                    f3=tfft(civa[0, 0, :], timestep),
                                    civa_noise=civa_noise[0],
                                    civa_r=civa_r,
                                    time=time)
                               )

for i in range(len(groups) - del_groups):
    p.line(time, X_ssa[i], legend_label='SSA {0}'.format(i + 1), line_color=color[i % len(color)],name='SSA {0}'.format(i + 1))

p_r.line('time', 'civa_noise', source=source_civa, legend_label="original", line_color="black", line_width=2)
p_r.line('time', 'civa_r', source=source_civa, legend_label="reconstructed", line_color="green", line_width=1)

# noise PC

for i in range(len(groups) - del_groups, len(groups) - del_groups + 5):
    p_n.line(time, X_ssa[i], legend_label='SSA {0}'.format(i + 1), line_color=color[i % len(color)],name='SSA {0}'.format(i + 1))

# compare representations time and frec
p_f.line('freq', 'f1', source=source_civa, legend_label="original+noise", line_color="black", line_width=2)
p_f.line('freq', 'f2', source=source_civa, legend_label="reconstructed", line_color="green", line_width=1)
p_f.line('freq', 'f3', source=source_civa, legend_label="original", line_color="blue", line_width=1)

set_legend_font(p)
set_legend_font(p_r)
set_legend_font(p_n)
set_legend_font(p_f)


# Example Slider
def update_plots(attr, old, new):
    global later
    now = timer_sec.perf_counter()

    if abs(now - later) > 2:
        later = now
        source_civa.data, X_ssa_ = get_example_reconstruction(new)

        for i in range(len(groups) - del_groups):
            remove_glyphs(p, ['SSA {0}'.format(i + 1)])
            p.line(time, X_ssa_[i] , legend_label='SSA {0}'.format(i + 1), line_color=color[i % len(color)],name='SSA {0}'.format(i + 1))
        for i in range(len(groups) - del_groups, len(groups) - del_groups + 5):
            remove_glyphs(p, ['SSA {0}'.format(i + 1)])
            p_n.line(time, X_ssa_[i],legend_label='SSA {0}'.format(i + 1), line_color=color[i % len(color)],name='SSA {0}'.format(i + 1))


slider_civa = Slider(title='CIVA n_ex', value=0, start=0, end=len(civa_noise[:, 0]) - 1, step=1)
slider_civa.on_change('value', update_plots)

curdoc().add_root(row(column(p, p_r, slider_civa), column(p_n, p_f, slider_civa)))

