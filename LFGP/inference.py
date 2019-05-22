import numpy as np
from blr import sample_regression_posterior
from utils import reshape_latent_curves
from metropolis import metropolis_sample
from tqdm import tqdm as tqdm


def gibbs_update_params(Y, model, chain_size=50, proposal_std=0.5):
    """
    Sample from model parameter posterior given Y and F

    Note: the model does not change at all
    """
    if model.F is None:
        model.update_conditional_latent(Y)
    F = model.F  # get current latent and condition on it for sampling
    loading, variance = sample_regression_posterior(Y, F, model.loading_prior_params, model.variance_prior_params)
    n, t, q, r = model.dims
    current_theta = model.theta
    theta = np.zeros(r)
    traces = np.zeros((r, chain_size))
    F_curves_list = reshape_latent_curves(F, n, t)
    for i, F_curves in enumerate(F_curves_list):
        l_trace, s_trace = metropolis_sample(chain_size, F_curves, model.x,
                                             prior_params=[model.length_prior_params[0],
                                                           model.length_prior_params[1], 1.0],
                                             proposal_scales=[proposal_std, proposal_std],
                                             l_start=current_theta[i])
        theta[i] = l_trace[-1]
        traces[i, :] = l_trace
    return F, loading, variance, theta, traces


def run_gibbs(Y, model, n_steps, chain_size, proposal_std, verbose=False):
    """
    Run Metropolis with-in Gibbs sampler on latent factor GP model using data Y

    Args
        Y: (numpy array) of shape [nt, q] and contains n stacked epochs of [t, q]
        model: latent factor GP model
        n_steps: (int) number of steps
        chain_size: (int) size of Metropolis chain at each iteration
        proposal_std: (float) standard deviation of Metropolis proposal distribution
        verbose: (bool) whether or not to print out MSE and length scale (for sanity check)
    """
    n, t, q, r = model.dims
    F_sample = np.zeros((n_steps, n * t, r))
    loading_sample = np.zeros((n_steps, r, q))
    variance_sample = np.zeros((n_steps, q))
    theta_sample = np.zeros((n_steps, r))
    traces_hist = np.zeros((n_steps, r, chain_size))
    mse_history = np.zeros(n_steps)

    for i in tqdm(range(n_steps)):
        F, loading, variance, theta, traces = gibbs_update_params(Y, model, chain_size, proposal_std)

        model.loading = loading  # update model parameters and predict
        model.variance = variance
        model.theta = theta
        Y_hat = model.predict()
        mse = np.mean((Y - Y_hat) ** 2)

        F_sample[i, :, :] = F  # save everything
        loading_sample[i, :, :] = loading
        variance_sample[i, :] = variance
        theta_sample[i, :] = theta
        traces_hist[i, :, :] = traces
        mse_history[i] = mse

        model.update_conditional_latent(Y)  # update model latent factors

        if verbose:
            print('Current MSE: {}'.format(mse))
            print('Current length scale: {}'.format(theta))

    return F_sample, loading_sample, variance_sample, theta_sample, traces_hist, mse_history
