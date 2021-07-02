'''

ALGORITHMS FOR FINITE RATE OF INNOVATION SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

Cite:
[1] M. Vetterli, P. Marziliano and T. Blu,
    "Sampling signals with finite rate of innovation," in 
    IEEE Transactions on Signal Processing, vol. 50, no. 6, pp. 1417-1428,
    June 2002, doi: 10.1109/TSP.2002.1003065.

[2] Pan, Hanjie, Thierry Blu, and Martin Vetterli.
    "Towards generalized FRI sampling with an application to source resolution in radioastronomy."
    IEEE Transactions on Signal Processing 65.4 (2016): 821-835.
    See codes: https://github.com/hanjiepan/FRI_pkg

[3] Simeoni, Matthieu, et al.
    "CPGD: Cadzow plug-and-play gradient descent for generalised FRI."
    IEEE Transactions on Signal Processing 69 (2020): 42-57.
    See codes: https://github.com/matthieumeo/pyoneer

'''

# %% LOAD LIBRARIES

import numpy as np

from scipy import linalg as lin

from astropy import units as uts
from astropy import coordinates as crd

import fri_operators

# %% UTILITY FUNCTIONS

def get_shifts(annihilating_filter, support):
    '''
    Get shifts from the annihilating filter

    '''

    roots = np.roots(np.flip(annihilating_filter, axis=0).reshape(-1))
    locations = crd.Angle(np.angle(roots) * uts.rad)
    locations = locations.wrap_at(2 * np.pi * uts.rad)
    return np.sort(support * locations.value.reshape(-1) / (2 * np.pi))

def get_weights(swce, locations, tau, M):
    '''
    Returns the weights of the FRI signal given the locations

    :param swce: sequence in sum-of-weighted-complex-exponentials form
    :param locations: estimated locations of the Dirac impulses
    :param tau: period of the signal
    :param M: dimension

    :return: weights of the FRI signal

    '''

    K = len(locations)
    swce_matrix = np.exp(-1j*2*np.pi*np.outer(np.arange(-M,M+1), locations)/tau)

    return (2*M + 1) * np.linalg.pinv(swce_matrix) @ swce

# %% PRONY

def cadzow_ls(x, M, P, rank, num_iter=20, tol=1e-12):
    '''
    Cadzow denoising of x with rank

    :param x: data vector
    :param M: dimension
    :param P: dimension
    :param rank: rank threshold
    :param num_iter: number of iterations
    :param tol: tolerance

    :returns: denoised data

    '''

    N = 2*M + 1
    for _ in range(num_iter):
       x = fri_operators.toeplitzification(x, M, P)
       x = fri_operators.low_rank_approximation(x, rank)
       x = fri_operators.pinv_toeplitzification(x, N, P)

    return x

def condat_hiribayashi():
    pass

def prony_tls(swce, model_order):
    '''
    High resolution spectral estimation (HRSE) using
    Annihilating Filter

    :param swce: input linear combination of sinusoids
    :param model_order: number of sinusoids

    :returns: annihilating filter

    '''

    N = swce.size
    M = int(N // 2)

    index_i = -M + model_order + np.arange(1, N - model_order + 1) - 1
    index_j = np.arange(1, model_order + 2) - 1
    index_ij = index_i[:, None] - index_j[None, :]
    conv_mtx = swce[index_ij + M]

    _, _, vh = lin.svd(conv_mtx, check_finite=False, full_matrices=False)
    annihilating_filter = vh[-1, :].conj()
    annihilating_filter = annihilating_filter.reshape(-1)

    return annihilating_filter

# %% GENERALISED FRI

# %% CPGD