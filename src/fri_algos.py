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

def get_weights(swce, locations, tau):
    '''
    Returns the weights of the FRI signal given the locations

    :param swce: sequence in sum-of-weighted-complex-exponentials form
    :param locations: estimated locations of the Dirac impulses
    :param tau: period of the signal
    :param M: dimension

    :return: weights of the FRI signal

    '''

    M = int(len(swce)//2)
    swce_matrix = np.exp(-1j*2*np.pi*np.outer(np.arange(-M,M+1), locations)/tau)

    return (2*M + 1) * np.linalg.pinv(swce_matrix) @ swce

# %% PRONY

def cadzow_ls(x, M, P, rank, rho=None, num_iter=20):
    '''
    Cadzow denoising of x with rank

    :param x: data vector
    :param M: dimension
    :param P: dimension
    :param rank: rank threshold
    :param rho: projection radius
    :param num_iter: number of iterations

    :returns: denoised data

    '''

    N = 2*M + 1
    if rho:
        for _ in range(num_iter):
            x = proj_l2_ball(x, rho)
            x = fri_operators.toeplitzification(x, M, P)
            x = fri_operators.low_rank_approximation(x, rank)
            x = fri_operators.pinv_toeplitzification(x, N, P)
    else:
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

def gen_fri(G, a, K, noise_level=0, max_ini=100, stop_cri='mse'):
    '''
    Alternating minimisation for Generalised FRI

    :param G: forward model
    :param a: measurements
    :param K: model order
    :param noise_level: noise level
    :param max_ini: maximun iterations
    :param stop_cri: stopping criterion

    :returns: fourier series coefficients
              minimum error
              annihilating filter
              initialisation

    See Pan et al. [2], for detailed codes

    '''
    compute_mse = (stop_cri == 'mse')
    M = G.shape[1]
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, a)

    max_iter = 50
    min_error = float('inf')
    # beta = linalg.solve(GtG, Gt_a)
    beta = lin.lstsq(G, a)[0]

    Tbeta = fri_operators.Tmtx(beta, K)
    rhs = np.concatenate((np.zeros(2 * M + 1), [1.]))
    rhs_bl = np.concatenate((Gt_a, np.zeros(M - K)))

    for ini in range(max_ini):
        c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = fri_operators.Rmtx(c, K, M)

        # first row of mtx_loop
        mtx_loop_first_row = np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T,
                                        np.zeros((K + 1, M)), c0[:, np.newaxis]))
        # last row of mtx_loop
        mtx_loop_last_row = np.hstack((c0[np.newaxis].conj(),
                                       np.zeros((1, 2 * M - K + 1))))

        for loop in range(max_iter):
            mtx_loop = np.vstack((mtx_loop_first_row,
                                  np.hstack((Tbeta, np.zeros((M - K, M - K)),
                                             -R_loop, np.zeros((M - K, 1)))),
                                  np.hstack((np.zeros((M, K + 1)), -R_loop.conj().T,
                                             GtG, np.zeros((M, 1)))),
                                  mtx_loop_last_row
                                  ))

            # matrix should be Hermitian symmetric
            mtx_loop += mtx_loop.conj().T
            mtx_loop *= 0.5
            # mtx_loop = (mtx_loop + mtx_loop.conj().T) / 2.

            c = lin.solve(mtx_loop, rhs)[:K + 1]

            R_loop = fri_operators.Rmtx(c, K, M)

            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((M - K, M - K))))
                                    ))

            # matrix should be Hermitian symmetric
            mtx_brecon += mtx_brecon.conj().T
            mtx_brecon *= 0.5
            # mtx_brecon = (mtx_brecon + mtx_brecon.conj().T) / 2.

            b_recon = lin.solve(mtx_brecon, rhs_bl)[:M]

            error_seq[loop] = lin.norm(a - np.dot(G, b_recon))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon
                c_opt = c
            if min_error < noise_level and compute_mse:
                break
        if min_error < noise_level and compute_mse:
            break

    return b_opt, min_error, c_opt, ini

# %% CPGD

def get_tau(G, P):
    '''
    Compute PGD parameter that guarantees convergence

    :param G: forward matrix
    :param P: model order

    :return: largest PGD parameter that guarantees convergence

    '''

    eig_vals, _ = np.linalg.eig(np.conj(G).T @ G)
    beta = np.max(eig_vals)

    lower_lim = np.abs((1-1/np.sqrt(P+1)) / beta)
    upper_lim = np.abs((1+1/np.sqrt(P+1)) / beta)

    return lower_lim + (upper_lim-lower_lim) * np.random.rand()

def proj_l2_ball(x, rho):
    '''
    Projection of x onto l2 ball with radius rho

    :param x: input data
    :param rho: radius

    :returns: projection

    '''
    
    norm = np.linalg.norm(x)
    if norm <= rho:
        return x
    else:
        return rho*x/norm

def cpgd(G, y, tau, P, rank, rho, init, num_iter=50):
    '''
    CPGD for FRI

    :param G: forward matrix
    :param y: measurements
    :param tau: parameter for PGD
    :param P: model order
    :param rank: rank threshold
    :param rho: radius of threshold
    :param init: initialisation

    :returns: fourier series coefficients

    '''

    _, N = G.shape
    M = int(N // 2)

    x = init
    for _ in range(num_iter):
        der = 2 * np.conj(G).T @ (G@x - y)
        z = x - tau * der
        x = cadzow_ls(z, M, P, rank, rho)

    return x