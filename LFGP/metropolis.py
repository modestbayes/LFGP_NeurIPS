import numpy as np
from scipy.stats import multivariate_normal
from utils import kernel_covariance, l_gamma_prior


def gp_marginal_likelihood(y, x, l, s):
    """
    Marginal likelihood of one Gaussian process (multivariate Normal)
    """
    t = y.shape[0]
    mu = np.repeat(0, t)
    cov = kernel_covariance(x, l, s)
    return multivariate_normal.pdf(y, mu, cov)


def propose(current, std):
    """
    Random walk proposal
    """
    value = -1
    while value < 0:
        value = np.random.normal(loc=current, scale=std, size=1)
    return value


def calculate_p_deprecated(l, s, Y, x, prior_params):
    """
    Calculate log prior and likelihood of n independent Gaussian processes (Y has shape [t, n])
    """
    a, b, scale = prior_params
    prior = l_gamma_prior(l, a, b)  # * s_half_cauchy_prior(s, scale)
    loglik = 0.0
    for j in range(Y.shape[1]):
        loglik += np.log(gp_marginal_likelihood(Y[:, j], x, l, s))  # independent observations
    return np.log(prior) + loglik


def calculate_p(l, s, Y, x, prior_params):
    a, b, scale = prior_params
    prior = l_gamma_prior(l, a, b)  # * s_half_cauchy_prior(s, scale)
    t, n = Y.shape
    cov = kernel_covariance(x, l, s)
    inverse = np.linalg.inv(cov)
    loglik = 0.0
    sign, logdet = np.linalg.slogdet(cov)
    constant = -0.5 * logdet - 0.5 * t * np.log(2 * np.pi)
    for j in range(Y.shape[1]):
        loglik += -0.5 * np.matmul(np.matmul(Y[:, j].reshape((1, t)), inverse), Y[:, j].reshape((t, 1)))[0][0] + constant
    return loglik + np.log(prior)


def accept_new(accept_prob):
    """
    Accept or reject proposed state
    """
    u = np.random.uniform(low=0.0, high=1.0, size=1)
    return u < accept_prob


def metropolis_update(l, s, p, Y, x, prior_params, proposal_scales):
    """
    Metropolis update step
    """
    l_new = propose(l, proposal_scales[0])
    s_new = 1.0

    p_new = calculate_p(l_new, s_new, Y, x, prior_params)

    if accept_new(np.exp(p_new - p)):
        return l_new, s_new, p_new
    else:
        return l, s, p


def metropolis_sample(n_iter, Y, x, prior_params, proposal_scales, l_start):
    """
    Run Metropolis chain for many iterations and return trace
    """
    l = l_start
    s = 1.0  # variance scale is fixed at 1
    p = calculate_p(l, s, Y, x, prior_params)

    l_trace = []
    s_trace = []
    for i in range(n_iter):
        l, s, p = metropolis_update(l, s, p, Y, x, prior_params, proposal_scales)
        l_trace.append(l)
        s_trace.append(s)

    return l_trace, s_trace
