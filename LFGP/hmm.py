import numpy as np
from hmmlearn import hmm
from scipy.linalg import logm


def predict_cov(Z, covs):
    """
    Predict covariance covariance Log-Euclidean vector time series.

    Args
        Z: (numpy array) predicted discrete latent states of length t
        covs: (numpy array) [k, n, n] covariance matrices for the k latent states
    """
    t = Z.shape[0]
    n = covs[0].shape[0]
    log_hat = np.zeros((t, int(0.5 * n * (n + 1))))
    for i in range(t):
        latent = Z[i]
        log_hat[i, :] = logm(covs[latent, :, :])[np.triu_indices(n)]
    return log_hat


def HMM(X, Y, k=4):
    """
    Hidden markov model for discrete covariance states.
    """
    model = hmm.GaussianHMM(n_components=k, covariance_type='full', n_iter=50, algorithm='map').fit(X)
    Z = model.predict(X)
    Y_hat = predict_cov(Z, model.covars_)
    return Z, Y_hat
