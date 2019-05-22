import numpy as np
import pickle
import os
from utils import sliding_window
from factorgp import FactorGP
from inference import run_gibbs


data_path = 'sim_data_new.npy'
output_dir = ''
scenario_idx = 0
scenario_name = 'square'
first = 0
last = 20


def prepare_data(sim_data, scenario, data_idx):
    """
    Prepare one set of simulation data.
    """
    length = 500
    start = data_idx * length * 4 + scenario * length
    end = start + length
    X = np.asarray(sim_data[start:end, :10])
    Y = np.asarray(sim_data[start:end, 10:65])
    F = np.asarray(sim_data[start:end, 65:69])
    return X, Y, F


sim_data = np.load(data_path)

for i in range(first, last):
    X, Y, F = prepare_data(sim_data, scenario_idx, i)
    log_series_hat = sliding_window(X, 50, 10)
    dims = [1, log_series_hat.shape[0], log_series_hat.shape[1], 4]
    model = FactorGP(dims)
    model.update_conditional_latent(log_series_hat)
    results = run_gibbs(log_series_hat, model, 100, 50, 0.5)
    with open(os.path.join(output_dir, '{name}_{i}.pkl'.format(name=scenario_name, i=i)), 'wb') as fp:
        pickle.dump(results, fp)
