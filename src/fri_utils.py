'''

UTILITY TOOLS FOR FRI SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

Cite:
[1] M. Vetterli, P. Marziliano and T. Blu,
    "Sampling signals with finite rate of innovation," in 
    IEEE Transactions on Signal Processing, vol. 50, no. 6, pp. 1417-1428,
    June 2002, doi: 10.1109/TSP.2002.1003065.

[2] R. Tur, Y. C. Eldar and Z. Friedman,
    "Innovation Rate Sampling of Pulse Streams With Application to Ultrasound Imaging," in
    IEEE Transactions on Signal Processing, vol. 59, no. 4, pp. 1827-1842,
    April 2011, doi: 10.1109/TSP.2011.2105480.

'''

# %% LOAD LIBRARIES

import os
import numpy as np

from numpy import matlib
from scipy import linalg as splin

from matplotlib import pyplot as plt

# %% HELPER FUNCTIONS

def periodise(signal, period):
    ''' Periodise signal with finite repetitions '''
    return np.matlib.repmat(signal, 1, period)

def add_noise(data, snr=None, sigma=None):
    '''
    Add white Gaussian noise to data according to given SNR or standard deviation

    :param data: input data vector
    :param snr: desired signal to noise ratio
    :param sigma: desired noise variance

    :returns: noisy data

    '''

    if snr:
        awgn = np.random.randn(len(data))
        awgn = awgn / np.linalg.norm(awgn) * np.linalg.norm(data) * 10 ** (-1.0*snr / 20.)

    elif sigma:
        awgn = np.random.normal(scale=sigma, loc=0, size=data.shape)

    return data + awgn

# %% PLOT TOOLS

def plot_diracs(tk, ak, ax=None, plot_colour='blue', line_width=2,
    marker_style='o', marker_size=4, line_style='-', legend_show=True,
    legend_loc='upper left', legend_label=None, title_text=None,
    xaxis_label=None, yaxis_label=None, xlimits=[0,1], ylimits=[-1,1],
    show=False, save=None):
    ''' Plots Diracs at tk, ak '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    markerline, stemlines, baseline = plt.stem(tk, ak, label=legend_label,
        linefmt=line_style)
    plt.setp(stemlines, linewidth=line_width, color=plot_colour)
    plt.setp(markerline, marker=marker_style, linewidth=line_width,
        markersize=marker_size, markerfacecolor=plot_colour, mec=plot_colour)
    plt.setp(baseline, linewidth=0)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper right', line_style='-', line_width=None,
    show=False, xlimits=[-2,2], ylimits=[-2,2], save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width, color=plot_colour,
        label=legend_label)
    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_hline(level=0, ax=None, line_colour='black', line_style='-',
    line_width=1, annotation=None, pos=(1,1)):
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.axhline(level, color=line_colour, linestyle=line_style, linewidth=line_width)
    if annotation:
        plt.annotate(annotation, xy=pos, color=line_colour)

def plot_mcerrors(x, y, ax=None, plot_colour='blue', line_width=2,
    marker_style='o', marker_size=4, line_style='-', legend_label=None,
    legend_loc='lower left', legend_show=True, title_text=None, dev_alpha=0.5,
    xaxis_label=None, yaxis_label=None, xlimits=[-30,30], ylimits=[1e-4, 1e2],
    show=False, save=None):
    ''' Plot x,y on semilogy '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    means = np.mean(y, axis=1)
    devs = np.std(y, axis=1)
    plt.semilogy(x, means, linestyle=line_style, linewidth=line_width,
        color=plot_colour, marker=marker_style, markersize=marker_size,
        label=legend_label)
    plt.fill_between(x, means-devs, means+devs, color=plot_colour,
        linewidth=0, alpha=dev_alpha)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return
# %% PULSE SHAPES

def cubic_Bspline(t, scale=1):
    '''
    Returns point evaluations of cubic b-splines

    :param t: evaluation points
    :param scale: temporal scale parameter

    :returns: point evaluations of cubic b-spline

    '''

    t = t*scale
    out = np.zeros(t.shape)
    
    idx = np.where(np.abs(t)<=1.0)
    out[idx] = 2.0/3.0 - np.abs(t[idx])**2 + np.abs(t[idx])**3/2.0

    idx = np.intersect1d(np.where(np.abs(t)<=2.0), np.where(np.abs(t)>1.0))
    out[idx] = (2.-np.abs(t[idx]))**3/6.

    return out

# %% SAMPLING KERNELS

def dirichlet(t, bandwidth, duration):
    '''
    Returns point evaluation of the Dirichlet kernel

    :param t: evaluation points
    :param bandwidth: bandwidth of the kernel
    :param duration: time support of the kernel

    :returns: point evaluations of the kernel

    '''

    numerator = np.sin(np.pi * bandwidth * t)
    denominator = bandwidth * duration * np.sin(np.pi * t / duration)
    
    idx = np.abs(denominator) < 1e-12
    
    numerator[idx] = np.cos(np.pi * bandwidth * t[idx])
    denominator[idx] = np.cos(np.pi * t[idx] / duration)
    return numerator / denominator

def sos(t, order, duration):
    '''
    Generate sum of sincs kernel in the time-domain

    :param t: evaluation points
    :param order: order of sincs
    :param duration: support of the kernel

    :returns: point evaluations of the sos kernel

    '''

    window = (np.abs(t)<duration)*1.0
    w = 2.0*np.pi/duration
    
    numerator = np.sin((order+0.5)*w*t)
    denominator = np.sin(w*t/2.)

    idx = np.where(np.abs(denominator) < 1e-12)
    numerator[idx] = np.cos((order+0.5)*w*t[idx])*w*t[idx]
    denominator[idx] = np.cos(w*t[idx]/2.)*0.5

    return (numerator/denominator) * window / (2.*order+1)