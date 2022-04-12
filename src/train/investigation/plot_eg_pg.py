from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# the random data
x_stead = np.load('STEAD_EG_STEAD.npy')
y_stead = np.load('STEAD_PG_STEAD.npy')

x_niigata = np.load('Niigata_EG_Niigata.npy')
y_niigata = np.load('Niigata_PG_Niigata.npy')

x_stead_filtered = np.load('STEAD_EG_STEAD_filtered.npy')
y_stead_filtered = np.load('STEAD_PG_STEAD_filtered.npy')

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x_stead, y_stead, c='#1f77b4', edgecolor='black')
axScatter.scatter(x_niigata, y_niigata, c='#ff0000', edgecolor='black')
axScatter.scatter(x_stead_filtered, y_stead_filtered, c='#800080', edgecolor='black')
axScatter.legend(['STEAD-broadband','Niigata','STEAD-filtered'])
# now determine nice limits by hand:
binwidth = 0.25


xymax = np.max([np.max(np.fabs(x_stead)), np.max(np.fabs(y_stead))])
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((0, lim))
axScatter.set_ylim((0, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x_stead, bins=bins)
axHistx.hist(x_stead_filtered, bins=bins, color='#800080')
axHistx.hist(x_niigata, bins=bins, color='#ff0000')

axHisty.hist(y_stead, bins=bins, orientation='horizontal')
axHisty.hist(y_stead_filtered, bins=bins,color='#800080', orientation='horizontal')
axHisty.hist(y_niigata, bins=bins,color='#ff0000', orientation='horizontal')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())
plt.savefig('list_of_gof_stead_niigata.png')