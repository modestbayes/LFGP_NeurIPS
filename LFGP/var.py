import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from simulation import sliding_window


def SW_PCA_VAR(X, Y, ws=50, r=2):
    """
    Vector auto-regression model.
    """
    log_series_hat = sliding_window(X, ws)
    pca = PCA(n_components=r)
    components = pca.fit_transform(log_series_hat)
    model = VAR(components)
    results = model.fit(2)
    fitted_components = np.zeros(components.shape)
    fitted_components[:2, :] = components[:2, :]
    fitted_components[2:, :] = results.fittedvalues
    Y_hat = pca.inverse_transform(fitted_components)
    return fitted_components, Y_hat
