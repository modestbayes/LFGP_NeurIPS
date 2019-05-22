import pystan
import numpy as np
from utils import reshape_latent_curves
from metropolis import metropolis_sample
from tqdm import tqdm as tqdm


def horseshoe_stan_model():
    horseshoe_code = """
    data {
      int<lower=1> N; // Number of data
      int<lower=1> M; // Number of covariates
      matrix[M, N] X;
      real y[N];
    }

    parameters {
      vector[M] beta_tilde;
      vector<lower=0>[M] lambda;
      real<lower=0> tau_tilde;
      real<lower=0> sigma;
    }

    transformed parameters {
      vector[M] beta = beta_tilde .* lambda * sigma * tau_tilde;
    }

    model {
      // tau ~ cauchy(0, sigma)
      // beta ~ normal(0, tau * lambda)

      beta_tilde ~ normal(0, 1);
      lambda ~ cauchy(0, 1);
      tau_tilde ~ cauchy(0, 0.1);

      sigma ~ normal(0, 0.1);

      y ~ normal(X' * beta, sigma);
    }
    """
    sm = pystan.StanModel(model_code=horseshoe_code)
    return sm


def horseshoe_regression_posterior(y, X, sm):
    n, m = X.shape
    horseshoe_dat = {'N': n, 'M': m, 'y': y, 'X': np.transpose(X)}
    fit = sm.sampling(data=horseshoe_dat, iter=1000, chains=1)
    beta_sample = fit.extract()['beta']
    sigma_sample = fit.extract()['sigma']
    return beta_sample, sigma_sample


def sample_regression_posterior(Y, F, sm):
    n, q = Y.shape
    r = F.shape[1]
    loading = np.zeros((r, q))
    variance = np.zeros(q)
    for j in range(q):
        y = Y[:, j]
        beta_sample, sigma_sample = horseshoe_regression_posterior(y, F, sm)
        loading[:, j] = beta_sample[-1, :]
        variance[j] = sigma_sample[-1]
    return loading, variance


def gibbs_update_params_hs(Y, model, sm, chain_size=50, proposal_std=0.5):
    """
    Sample from model parameter posterior given Y and F

    Note: the model does not change at all
    """
    if model.F is None:
        model.update_conditional_latent(Y)
    F = model.F  # get current latent and condition on it for sampling
    loading, variance = sample_regression_posterior(Y, F, sm)
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


def run_gibbs_hs(Y, model, sm, n_steps, chain_size, proposal_std, verbose=False):
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
        F, loading, variance, theta, traces = gibbs_update_params_hs(Y, model, sm, chain_size, proposal_std)

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
