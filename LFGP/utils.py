import numpy as np
from scipy.stats import gamma, cauchy, norm
from scipy.linalg import logm


def l_gamma_prior(l, a, b):
    """
    Gamma prior on length scale
    """
    return gamma.pdf(l, a=a, scale=1/b)


def s_half_cauchy_prior(s, scale):
    return 2 * cauchy.pdf(s, loc=0, scale=scale)


def kernel_covariance(x, l, s, noise=1e-6):
    """
    Covariance matrix with squared exponential kernel
    """
    t = x.shape[0]
    cov_sample = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            cov_sample[i, j] = s ** 2 * np.exp(-(x[i] - x[j]) ** 2 / (2 * l ** 2))
    cov_sample += np.eye(t) * noise  # add noise for numerical stability
    return cov_sample


def reshape_latent_curves(F, n, t):
    """
    Turn latent factors F of shape [nt, r] into a list of r factors of shape [t, n]
    """
    r = F.shape[1]
    F_curves_list = []
    for j in range(r):
        F_curves = np.zeros((t, n))
        for i in range(n):
            F_curves[:, i] = F[(i * t):(i * t + t), j]
        F_curves_list.append(F_curves)
    return F_curves_list


def sliding_window(time_series, size=50, stride=1):
    """
    Calculate sliding window covariance Log-Euclidean vector time series.

    Args
        time_series: (numpy array) [t, n] t observations in time of n dimensional data
        size: (int) sliding window size
        stride: (int) sliding step size
    """
    t, n = time_series.shape
    # cov_series = np.zeros((t - size, n, n))
    log_series = np.zeros((int((t - size) / stride), int(0.5 * n * (n + 1))))
    for i in range(int((t - size) / stride)):
        window = time_series[(i * stride):(i * stride + size), :]
        cov = np.cov(window, rowvar=False)
        # cov_series[i, :, :] = cov
        log_series[i, :] = logm(cov)[np.triu_indices(n)]
    return log_series


def gaussian_taper(cov_series, i, size, z):
    """
    Gaussian tapering of estimated covariance series.

    Args
        cov_series: (numpy array) [n, k, k] estimated covariance series
        i: (int) current index
        size: (int) sliding window size
        z: (float) Gaussian shape in terms of z-score
    """
    n = cov_series.shape[0]
    k = cov_series.shape[1]
    start_index = max(0, i - int(size * 0.5))
    end_index = min(n, i + int(size * 0.5))
    cov_window = cov_series[start_index:end_index, :, :]
    left_width = min(i, int(size * 0.5))
    right_width = min(n - i, int(size * 0.5))
    weights = norm.pdf(np.arange(-size * 0.5, size * 0.5, 1), loc=0, scale=size * z)
    weight_window = weights[(int(size * 0.5) - left_width):(int(size * 0.5) + right_width)]
    weight_window = weight_window / np.sum(weight_window)
    cov_hat = np.zeros((k, k))
    for j in range(end_index - start_index):
        cov_hat += cov_window[j, :, :] * weight_window[j]
    return cov_hat


def tapered_sliding_window(time_series, size=50, stride=1, z=0.2):
    """
    Calculate sliding window covariance Log-Euclidean vector time series with Gaussian tapering.

    Args
        time_series: (numpy array) [t, n] t observations in time of n dimensional data
        size: (int) sliding window size
        stride: (int) sliding step size
        z: (float) Gaussian shape in terms of z-score
    """
    t, n = time_series.shape
    cov_series = np.zeros((int((t - size) / stride), n, n))
    log_series = np.zeros((int((t - size) / stride), int(0.5 * n * (n + 1))))
    for i in range(int((t - size) / stride)):
        window = time_series[(i * stride):(i * stride + size), :]
        cov = np.cov(window, rowvar=False)
        cov_series[i, :, :] = cov
    for i in range(int((t - size) / stride)):
        cov_hat = gaussian_taper(cov_series, i, size, z)
        log_series[i, :] = logm(cov_hat)[np.triu_indices(n)]
    return log_series


def create_symmetric_matrix(upper, dim=5):
    """
    Create a symmetric matrix with elements in the upper triangle.

    Args
        upper: (numpy array) vector with concatenated rows in the upper triangle
        dim: (int) square matrix dimension
    """
    mat = np.zeros((dim, dim))
    current = 0
    for i in range(dim):  # fill in the upper triangle
        end = current + dim - i
        mat[i, i:] = upper[current:end]
        current = end
    for i in range(1, dim):  # fill in the lower triangle
        for j in range(i):
            mat[i, j] = mat[j, i]
    return mat


def prepare_data(sim_data, scenario, data_idx):
    """
    Organize the simulated noisy time series, true covariance series and latent factors.
    """
    length = 500
    start = data_idx * length * 4 + scenario * length
    end = start + length
    X = np.asarray(sim_data[start:end, :10])
    Y = np.asarray(sim_data[start:end, 10:65])
    F = np.asarray(sim_data[start:end, 65:69])
    return X, Y, F
