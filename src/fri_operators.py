'''

OPERATORS FOR FINITE RATE OF INNOVATION SIGNAL PROCESSING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

Cite:
[1] T. Blu, P. Dragotti, M. Vetterli, P. Marziliano and L. Coulot,
    "Sparse Sampling of Signal Innovations,"
    in IEEE Signal Processing Magazine, vol. 25, no. 2, pp. 31-40,
    March 2008, doi: 10.1109/MSP.2007.914998.

[2] Simeoni, Matthieu, et al.
    "CPGD: Cadzow plug-and-play gradient descent for generalised FRI."
    IEEE Transactions on Signal Processing 69 (2020): 42-57.
    See codes: https://github.com/matthieumeo/pyoneer

'''

# %% LOAD LIBRARIES

import numpy as np

from scipy import linalg as splin

# %% SAMPLING MODELS

def forward_matrix(time_samples, model_order, period, mode=None):
    '''
    Constructs matrix of forward transform to Fourier series coefficients
    given time samples, period and model order.

    In the special case under critical uniform sampling,
    return the DFT matrix

    :param time_samples: locations of samples
    :param model_order: number of Fourier series coefficients
    :param period: period of the FRI signal

    :returns: forward matrix

    '''

    K = model_order
    if mode == None:
        return np.exp(1j*2*np.pi*np.outer(time_samples, np.arange(-K,K+1))/period)

    elif mode == 'iftem':
        num_samples = len(time_samples)
        F1 = np.exp(1j*2*np.pi*np.outer(time_samples[1::], np.arange(-K,K+1))/period)
        F2 = np.exp(1j*2*np.pi*np.outer(time_samples[0:num_samples-1], np.arange(-K,K+1))/period)
        F = F1 - F2
        F[:, K] = time_samples[1::] - time_samples[0:num_samples-1]

        scale_matrix = period/(1j*2*np.pi*np.arange(-K,K+1))
        scale_matrix[K] = 1
        scale_matrix = np.diag(scale_matrix)

        return np.matmul(F, scale_matrix)

# %% TOEPLITZIFICATION OPERATORS

def toeplitzification(x, M, P):
    '''
    Toeplitzification of an odd length vector
    
    :param x: generator

    :returns: toeplitzification

    '''

    N = 2*M+1
    index_i = -M + P + np.arange(1, N - P + 1) - 1
    index_j = np.arange(1, P + 2) - 1
    index_ij = index_i[:, None] - index_j[None, :]
    Tp_x = x[index_ij + M]
    return Tp_x

def adj_toeplitzification(x, N, P):
    '''
    Adjoint of Toeplitzification
    
    :param x: toeplitz matrix

    :return: adjoint of toeplitz matrix
    
    '''

    offsets = -(np.arange(1, N + 1) - 1 - P)
    out = np.zeros(shape=(N,), dtype=x.dtype)
    for (i, m) in enumerate(offsets):
        out[i] = np.sum(np.diagonal(x, offset=m))
    
    return out

def toep_gram(P, N):
    '''
    Gram matrix of toeplitzification
    
    :param P: dimension
    :param N: dimension

    :return: Diagonal elements of gram matrix

    '''

    weights = np.ones(shape=(N,)) * (P + 1)
    weights[:P] = np.arange(1, P + 1)
    weights[N - P:] = np.flip(np.arange(1, P + 1), axis=0)
    
    return weights

def pinv_toeplitzification(x, N, P):
    '''
    Pseudoinverse of toeplitzification operator

    :param x: generator
    :param N: dimension
    :param P: dimension

    :return: generator
    
    '''

    return adj_toeplitzification(x, N, P) / toep_gram(P, N)

# %% CADZOW DENOISING

def low_rank_approximation(data, rank):
    '''
    Low-rank approxiation of data with rank

    :param data: input matrix
    :param rank: target rank

    '''
    
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    return (u[:, :rank] * s[None, :rank]) @ vh[:rank, :]