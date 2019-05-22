library(factorstochvol)
library(tidyverse)

sim_dat <- read_csv("simulation_data.csv")
sim_dat <- sim_dat[, -1]
colnames(sim_dat)[1] <- "X1"
dim(sim_dat)

ggplot(sim_dat %>% filter(seed == 1, scenario == 1)) + 
  geom_line(aes(x = t, y = X1), alpha = 0.3, size = 0.5) + 
  geom_line(aes(x = t, y = F1), color = "blue", size = 1.3) + 
  geom_line(aes(x = t, y = F2), color = "orange", size = 1.3) + 
  geom_line(aes(x = t, y = Y1), linetype = 2, size = 1.3) +
  theme_classic()

r <- sim_dat$r[1]
p <- sim_dat$p[1]
q <- sim_dat$q[1]
n_time <- 1000

mse_dat <- matrix(nrow = 100, ncol = 5)

# Run simulations for each of the four scenarios
# Store all simulation results in mse_dat
set.seed(1234)
for (s in 1:4) {
  mse <- c()
  cat(s, "\n")
  for (i in 1:100) {
    if (i %% 10 == 0) {
      cat(i / 10)
    }
    X <- sim_dat %>% filter(seed == i, scenario == s) %>% select(contains("X"))
    Y <- sim_dat %>% filter(seed == i, scenario == s) %>% select(contains("Y"))
    res <- fsvsample(as.matrix(X), factors = r, draws = 200, burnin = 50, runningstore = 6, quiet = T)
    
    cov_series <- array(0, dim = c(n_time, p, p))
    log_cov_series <- array(dim = c(n_time, q))
    for (k in 1:n_time) {
      cov_series[k, , ][lower.tri(cov_series[k, , ], diag = TRUE)] <- res$runningstore$cov[k, , 1]
      cov_series[k, , ] <- cov_series[k, , ] + t(cov_series[k, , ]) - diag(diag(cov_series[k, , ]))
      log_cov_series[k, ] <- expm::logm(cov_series[k, , ])[lower.tri(cov_series[k, , ], diag = TRUE)]
    }
    mse[i] <- mean((as.matrix(Y) - log_cov_series)^2)
  }
  mse_dat[, s] <- mse
  cat("\n")
}

# Histograms of simulation MSE
par(mfrow = c(4, 1))
for (j in 1:4) {
  hist(mse_dat[, j], breaks = 50)
  print(mean(mse_dat[, j]))
  print(median(mse_dat[, j]))
}
