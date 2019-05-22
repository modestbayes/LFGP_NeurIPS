# Modeling Dynamic Functional Connectivity with Latent Factor Gaussian Processes

Studying dynamic functional connectivity can lead to better understanding of the brain. Modeling dynamic functional connectivity is met with challenges of high dimensionality and noisy data. The common models used in neuroimaging studies such as principal component analysis and hidden Markov models are either non-probabilistic or inflexible. We present a probabilistic model to learn low-dimensional latent connectivity dynamics with Gaussian processes.

## LFGP

Source code for the LFGP model:

`factor_gp.py` LFGP model class

`blr.py` Bayesian linear regression with conjugate prior for factor loadings

`inference.py` Gibbs sampling algorithm for posterior inference

`metropolis.py` Metropolis random walk (used within Gibbs) for GP hyper-parameters

`mvn.py` Multivariate Gaussian conditional distribution and covariance decomposition


## final_plots

High resolution plots
