library(tidyverse)
library(magrittr)
library(survival)
library(glmnet)

n_bins <- 30

weights_gen <-
  function(data = simData,
           n_feats = 2,
           n_bins = n_bins,
           c = 1) {
    d <- n_feats * n_bins
    V <- l_c_n <- w <- matrix(NA, nrow = n_feats, ncol = n_bins)
    for (j in 1 : n_feats) {
      V[j, ] <- apply(data[ , colnames(data)[grepl(paste0("X", j, "_bin_"), colnames(data))]],
                      2, sum) / nrow(data)
      l_c_n[j, ] <- 2 * log(log((2 * nrow(data) * V[j, ] + 18.66 * exp(1) * (c + log(n_feats + d))) / (8)))
      w[j , ] <- 11.32 * sqrt((c + log(n_feats + d) + l_c_n[j , ]) / (nrow(data)) * V[j , ]) +
        18.62 * (c + 1 + log(n_feats + d) + l_c_n[j , ]) / (nrow(data))
    }
    
    return(w)
  }

m1_metric <-
  function(c_true, c_est) {
    m1 <- rep(NA, nrow(c_true))
    for (r in 1 : nrow(c_true)) {
      m1_tmp1 <- rep(NA, length(c_est[r, ]))
      for (i in 1 : length(c_est[r, ])) {
        m1_tmp1[i] <- min(abs(c_est[r, ][i] - c_true[r, ]))
      }
      m1_tmp2 <- rep(NA, length(c_true[r, ]))
      for (i in 1 : length(c_true[r, ])) {
        m1_tmp2[i] <- min(abs(c_true[r, ][i] - c_est[r, ]))
      }
      m1[r] <- max(m1_tmp1, m1_tmp2)
    }
    return(mean(m1))
  }

simData <- read.csv("binacox/simData_1000.csv")[ , -1]


for (i in 1 : n_bins) {
  simData %<>%
    mutate(temp1 = ifelse(X1 >= quantile(X1, i / (n_bins + 1)), 1, 0),
           temp2 = ifelse(X2 >= quantile(X2, i / (n_bins + 1)), 1, 0))
  if (i < 10){
    colnames(simData)[colnames(simData) == "temp1"] <- paste0("X1_bin_0", i)
    colnames(simData)[colnames(simData) == "temp2"] <- paste0("X2_bin_0", i)
  }
  else {
    colnames(simData)[colnames(simData) == "temp1"] <- paste0("X1_bin_", i)
    colnames(simData)[colnames(simData) == "temp2"] <- paste0("X2_bin_", i)
  }
}
simData_mod <-
  simData %>%
  dplyr::select(Y, delta) %>%
  bind_cols(simData[ , colnames(simData)[grepl("X1_bin_", colnames(simData))]]) %>%
  bind_cols(simData[ , colnames(simData)[grepl("X2_bin_", colnames(simData))]])

weights_simData <-
  weights_gen(data = simData,
              n_feats = 2,
              n_bins = n_bins,
              c = 10)
weights_simData[1, ]



X_mat <-
  model.matrix( ~ . -1,
                data = simData_mod[ , colnames(simData_mod)[grepl("_bin_", colnames(simData_mod))]])


glm_cox_bussy <- glmnet(x = X_mat, y = Surv(simData_mod$Y, simData_mod$delta),
                      family = "cox", lambda = 1,
                      penalty.factor = c(weights_simData[1, ], weights_simData[2, ]))
names(glm_cox_bussy$beta[ , 1][glm_cox_bussy$beta[, 1] != 0])



glm_cv <- cv.glmnet(x = X_mat, y = Surv(simData_mod$Y,
                                        simData_mod$delta),
                    family = "cox")
glm_cv$lambda.1se
plot(glm_cv)
glm_cox <- glmnet(x = X_mat, y = Surv(simData_mod$Y, simData_mod$delta),
                      family = "cox", lambda = glm_cv$lambda.1se * 3.5)
names(glm_cox$beta[ , 1][glm_cox$beta[, 1] != 0])

X1_cuts <- quantile(simData$X1, probs = c(11, 23) / (n_bins + 1))
X2_cuts <- quantile(simData$X2, probs = c(9, 23) / (n_bins + 1))

m1_metric(c_true = matrix(c(-0.38828854, 0.57771615, -0.61267879, 0.60985693), ncol = 2),
          c_est = matrix(c(quantile(simData$X1, probs = c(11, 23) / (n_bins + 1)),
                           quantile(simData$X2, probs = c(9, 23) / (n_bins + 1))), ncol = 2))

X1_cuts_bussy <- c(-0.37379203, 0.58799982)
X2_cuts_bussy <- c(-0.61961702, 0.6224972)
m1_metric(c_true = matrix(c(-0.38828854, 0.57771615, -0.61267879, 0.60985693), ncol = 2),
          c_est = matrix(c(X1_cuts_bussy, X2_cuts_bussy), ncol = 2))


# true cuts
simData %>%
  mutate(X1_bin = factor(ifelse(X1 <= -0.38828854, 0,
                                ifelse(X1 <= 0.57771615, 1, 2))),
         X2_bin = factor(ifelse(X2 <= -0.61267879, 0,
                                ifelse(X2 <= 0.60985693, 1, 2)))) %>%
  coxph(Surv(Y, delta) ~ X1_bin + X2_bin, data = .) %>% AIC
  
# our cuts
simData %>%
  mutate(X1_bin = factor(ifelse(X1 <= X1_cuts[1], 0,
                                ifelse(X1 <= X1_cuts[2], 1, 2))),
         X2_bin = factor(ifelse(X2 <= X2_cuts[1], 0,
                                ifelse(X2 <= X2_cuts[2], 1, 2)))) %>%
  coxph(Surv(Y, delta) ~ X1_bin + X2_bin, data = .) %>% AIC

# Bussy's cuts
simData %>%
  mutate(X1_bin = factor(ifelse(X1 <= X1_cuts_bussy[1], 0,
                                ifelse(X1 <= X1_cuts_bussy[2], 1, 2))),
         X2_bin = factor(ifelse(X2 <= X2_cuts_bussy[1], 0,
                                ifelse(X2 <= X2_cuts_bussy[2], 1, 2)))) %>%
  coxph(Surv(Y, delta) ~ X1_bin + X2_bin, data = .) %>% AIC



