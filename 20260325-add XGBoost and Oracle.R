# ==============================================================================
# 回归型近端因果推断 (PCI) 模拟研究脚本 (完整扩展版)
# 新增点：
# 1. 引入 Oracle 方法：作为理论天花板，拥有真实的非线性函数 DGP 视角
# 2. 引入 XGBoost 方法：结合交叉拟合，测试前沿梯度提升树在 PCI 中的表现
# ==============================================================================

library(dplyr)
library(parallel)
library(mgcv)      # GAM
library(splines)   # RCS
library(ranger)    # RF
library(xgboost)   # XGBoost (新增)
library(sandwich)  # 稳健标准误 
library(lmtest)    # 配合 sandwich

# ------------------------------------------------------------------------------
# 1. 数据生成机制 (DGP)
# ------------------------------------------------------------------------------
generate_master_data <- function(n = 1000, scenario = "1_NonLinear", 
                                 gamma_z = 1.5, gamma_w = 1.5) {
  X1 <- rnorm(n, 0, 1)
  X2 <- rnorm(n, 0, 1)
  X3 <- rbinom(n, 1, 0.5)
  U  <- rnorm(n, 0, 1)
  beta_true <- 2.0
  
  Z <- gamma_z * U + 0.1*X1 + 0.1*X2 - 0.1*X3 + rnorm(n, 0, 0.5)
  
  if (scenario == "0_Linear") {
    prob_A  <- plogis(-1.5 + 0.5*U + 0.4*X1 - 0.2*X2 + 0.5*X3)
    W_clean <- gamma_w*U + 0.2*X1 + 0.2*X2 - 0.2*X3
    y_noA   <- 1.5*U + 0.8*X1 + 0.5*X2 + 0.5*X3
  } else if (scenario == "1_NonLinear") {
    prob_A  <- plogis(-1.5 + 0.5*U + 0.8*sin(1.5*X1) - 0.2*X2 + 0.5*X3)
    W_clean <- gamma_w*U + 0.5*sin(1.5*X1) + 0.2*X2 - 0.2*X3
    y_noA   <- 1.5*U + 2.0*cos(1.5*X1) + 0.5*X2 + 0.5*X3
  } else if (scenario == "2_Interaction") {
    prob_A  <- plogis(-1.5 + 0.5*U + 0.8*sin(1.5*X1)*X3 - 0.2*X2 + 0.5*X3)
    W_clean <- gamma_w*U + 0.5*sin(1.5*X1)*X3 + 0.2*X2 - 0.2*X3
    y_noA   <- 1.5*U + 2.0*cos(1.5*X1)*X3 + 0.5*X2 + 0.5*X3
  }
  
  A <- rbinom(n, 1, prob_A)
  Y <- beta_true * A + y_noA + rnorm(n, 0, 1)
  W <- W_clean + rnorm(n, 0, 0.5)
  
  return(data.frame(Y=Y, A=A, Z=Z, W=W, X1=X1, X2=X2, X3=X3, W_clean=W_clean))
}

# ------------------------------------------------------------------------------
# 2. 估计器实现 
# ------------------------------------------------------------------------------

# 【新增】方法 0: Oracle (先知模型，完全知道真实的 DGP 公式)
est_oracle <- function(df, sc, return_mod = FALSE) {
  if (sc == "0_Linear") {
    f1 <- lm(W ~ A + Z + X1 + X2 + X3, data = df)
    df$hatW <- predict(f1, newdata = df)
    f2 <- lm(Y ~ A + hatW + X1 + X2 + X3, data = df)
  } else if (sc == "1_NonLinear") {
    # 完美知晓 X1 的 sin 和 cos 效应
    f1 <- lm(W ~ A + Z + sin(1.5*X1) + X2 + X3, data = df)
    df$hatW <- predict(f1, newdata = df)
    f2 <- lm(Y ~ A + hatW + cos(1.5*X1) + X2 + X3, data = df)
  } else if (sc == "2_Interaction") {
    # 完美知晓非线性交互
    f1 <- lm(W ~ A + Z + I(sin(1.5*X1)*X3) + X2 + X3, data = df)
    df$hatW <- predict(f1, newdata = df)
    f2 <- lm(Y ~ A + hatW + I(cos(1.5*X1)*X3) + X2 + X3, data = df)
  }
  
  est_val <- unname(coef(f2)["A"])
  mse_val <- mean((df$hatW - df$W_clean)^2)
  if(return_mod) return(list(est = est_val, mse_s = mse_val, mod = f2))
  c(est = est_val, mse_s = mse_val)
}

# 方法 A: 标准线性 PCI
est_linear <- function(df, return_mod = FALSE) {
  f1 <- lm(W ~ A + Z + X1 + X2 + X3, data = df)
  df$hatW <- predict(f1, newdata = df)
  f2 <- lm(Y ~ A + hatW + X1 + X2 + X3, data = df)
  
  est_val <- unname(coef(f2)["A"])
  mse_val <- mean((df$hatW - df$W_clean)^2)
  if(return_mod) return(list(est = est_val, mse_s = mse_val, mod = f2))
  c(est = est_val, mse_s = mse_val)
}

# 方法 B: GAM PCI + 交叉拟合
est_gam_cf <- function(df, K = 5, return_mod = FALSE) {
  n <- nrow(df); folds <- sample(rep(1:K, length.out = n))
  hatW <- numeric(n)
  for (k in 1:K) {
    tr <- which(folds != k); val <- which(folds == k)
    f1 <- gam(W ~ A + Z + s(X1) + s(X2) + factor(X3), data = df[tr,])
    hatW[val] <- predict(f1, newdata = df[val,])
  }
  df$hatW <- hatW
  f2 <- gam(Y ~ A + hatW + s(X1) + s(X2) + factor(X3), data = df)
  
  est_val <- unname(coef(f2)["A"])
  mse_val <- mean((df$hatW - df$W_clean)^2)
  if(return_mod) return(list(est = est_val, mse_s = mse_val, mod = f2))
  c(est = est_val, mse_s = mse_val)
}

# 方法 C: RCS PCI 
est_rcs <- function(df, df_s = 4, return_mod = FALSE) {
  f1 <- lm(W ~ A + Z + ns(X1, df=df_s) + ns(X2, df=df_s) + X3, data = df)
  df$hatW <- predict(f1, newdata = df)
  f2 <- lm(Y ~ A + hatW + ns(X1, df=df_s) + ns(X2, df=df_s) + X3, data = df)
  
  est_val <- unname(coef(f2)["A"])
  mse_val <- mean((df$hatW - df$W_clean)^2)
  if(return_mod) return(list(est = est_val, mse_s = mse_val, mod = f2))
  c(est = est_val, mse_s = mse_val)
}

# 方法 D: RF PCI + 交叉拟合
est_rf_cf <- function(df, K = 5, return_mod = FALSE) {
  n <- nrow(df); folds <- sample(rep(1:K, length.out = n))
  hatW <- numeric(n)
  for (k in 1:K) {
    tr <- which(folds != k); val <- which(folds == k)
    f1 <- ranger(W ~ A + Z + X1 + X2 + X3, data = df[tr,], num.trees = 300)
    hatW[val] <- predict(f1, data = df[val,])$predictions
  }
  df$hatW <- hatW
  f2 <- lm(Y ~ A + hatW + ns(X1, df=4) + ns(X2, df=4) + X3, data = df)
  
  est_val <- unname(coef(f2)["A"])
  mse_val <- mean((df$hatW - df$W_clean)^2)
  if(return_mod) return(list(est = est_val, mse_s = mse_val, mod = f2))
  c(est = est_val, mse_s = mse_val)
}

# 【新增】方法 E: XGBoost PCI + 交叉拟合
est_xgb_cf <- function(df, K = 5, return_mod = FALSE) {
  n <- nrow(df); folds <- sample(rep(1:K, length.out = n))
  hatW <- numeric(n)
  
  # XGBoost 必须使用矩阵格式
  features <- as.matrix(df[, c("A", "Z", "X1", "X2", "X3")])
  target <- df$W
  
  # 设置比较稳健的超参数，防止小样本下深度过拟合
  params <- list(booster = "gbtree", objective = "reg:squarederror", 
                 max_depth = 3, eta = 0.1)
  
  for (k in 1:K) {
    tr <- which(folds != k); val <- which(folds == k)
    dtrain <- xgb.DMatrix(data = features[tr, ], label = target[tr])
    dval <- xgb.DMatrix(data = features[val, ])
    
    f1 <- xgb.train(params = params, data = dtrain, nrounds = 50, verbose = 0)
    hatW[val] <- predict(f1, dval)
  }
  df$hatW <- hatW
  # 第二阶段维持统一的盲测样条控制，确保系数提取得绝对线性
  f2 <- lm(Y ~ A + hatW + ns(X1, df=4) + ns(X2, df=4) + X3, data = df)
  
  est_val <- unname(coef(f2)["A"])
  mse_val <- mean((df$hatW - df$W_clean)^2)
  if(return_mod) return(list(est = est_val, mse_s = mse_val, mod = f2))
  c(est = est_val, mse_s = mse_val)
}


# ------------------------------------------------------------------------------
# 3. 标准误 (SE) 计算模块
# ------------------------------------------------------------------------------
get_boot_se <- function(df, method_func, R = 50) {
  boot_ests <- replicate(R, {
    idx <- sample(1:nrow(df), replace = TRUE)
    method_func(df[idx, ])["est"]
  })
  return(sd(boot_ests))
}

run_one <- function(n, sc, method_name, gamma_z, gamma_w, calc_se = FALSE, boot_R = 50) {
  df <- generate_master_data(n, sc, gamma_z, gamma_w)
  
  if (method_name == "Oracle") {
    # Oracle 属于参数模型，可以用三明治方差极速计算 SE
    res_tmp <- est_oracle(df, sc, return_mod = TRUE)
    est_val <- res_tmp$est
    mse_val <- res_tmp$mse_s
    se_val  <- ifelse(calc_se, unname(sqrt(diag(vcovHC(res_tmp$mod, type="HC3")))["A"]), NA_real_)
  } else if (method_name == "Linear") {
    res_tmp <- est_linear(df, return_mod = TRUE)
    est_val <- res_tmp$est
    mse_val <- res_tmp$mse_s
    se_val  <- ifelse(calc_se, unname(sqrt(diag(vcovHC(res_tmp$mod, type="HC3")))["A"]), NA_real_)
  } else if (method_name == "RCS") {
    res_tmp <- est_rcs(df, return_mod = TRUE)
    est_val <- res_tmp$est
    mse_val <- res_tmp$mse_s
    se_val  <- ifelse(calc_se, unname(sqrt(diag(vcovHC(res_tmp$mod, type="HC3")))["A"]), NA_real_)
  } else if (method_name == "GAM_CF") {
    res_tmp <- est_gam_cf(df) 
    est_val <- res_tmp["est"]
    mse_val <- res_tmp["mse_s"]
    se_val  <- ifelse(calc_se, get_boot_se(df, est_gam_cf, R = boot_R), NA_real_)
  } else if (method_name == "RF_CF") {
    res_tmp <- est_rf_cf(df)  
    est_val <- res_tmp["est"]
    mse_val <- res_tmp["mse_s"]
    se_val  <- ifelse(calc_se, get_boot_se(df, est_rf_cf, R = boot_R), NA_real_)
  } else if (method_name == "XGB_CF") {
    res_tmp <- est_xgb_cf(df)  
    est_val <- res_tmp["est"]
    mse_val <- res_tmp["mse_s"]
    se_val  <- ifelse(calc_se, get_boot_se(df, est_xgb_cf, R = boot_R), NA_real_)
  }
  
  return(data.frame(Method = method_name, 
                    Est = unname(est_val), 
                    MSE_S = unname(mse_val), 
                    SE = unname(se_val),
                    gamma_z = gamma_z, 
                    gamma_w = gamma_w))
}

# ------------------------------------------------------------------------------
# 4. 执行模拟 wrapper
# ------------------------------------------------------------------------------
run_full_simulation <- function(n_sim = 100, n = 1000, scenario = "2_Interaction", calc_se = FALSE) {
  # 将 Oracle 和 XGB_CF 加入执行队列
  methods <- c("Oracle", "Linear", "RCS", "GAM_CF", "RF_CF", "XGB_CF")
  
  gamma_grid <- expand.grid(gamma_z = c(0.3, 1.5), gamma_w = c(1.5)) 
  
  results <- mclapply(1:n_sim, function(i) {
    set.seed(i + 2026)
    iter_res <- list()
    for(m in methods) {
      for(j in 1:nrow(gamma_grid)) {
        res <- run_one(n, scenario, m, gamma_grid$gamma_z[j], gamma_grid$gamma_w[j], calc_se = calc_se)
        iter_res[[length(iter_res)+1]] <- res
      }
    }
    bind_rows(iter_res)
  }, mc.cores = 1) 
  
  beta_true <- 2.0
  final_tab <- bind_rows(results) %>%
    mutate(Bias = Est - beta_true) %>%
    group_by(Method, gamma_z, gamma_w) %>%
    summarise(
      Mean_Est = mean(Est),
      Abs_Bias = abs(mean(Bias)),
      RMSE = sqrt(mean(Bias^2)),
      Mean_MSE_S = mean(MSE_S),
      .groups = "drop"
    ) %>%
    arrange(gamma_z, RMSE)
  
  return(final_tab)
}

# ------------------------------------------------------------------------------
# 5. 测试运行
# ------------------------------------------------------------------------------
cat("开始运行包含 Oracle 和 XGBoost 的测试...\n")
test_res <- run_full_simulation(n_sim = 200, n = 1000, scenario = "2_Interaction", calc_se = FALSE)
print(test_res)
# A tibble: 12 × 7
# Method gamma_z gamma_w Mean_Est Abs_Bias   RMSE Mean_MSE_S
# <chr>    <dbl>   <dbl>    <dbl>    <dbl>  <dbl>      <dbl>
#   1 Oracle     0.3     1.5     1.96   0.0365 0.0993      1.60 
# 2 GAM_CF     0.3     1.5     1.96   0.0380 0.105       1.65 
# 3 RCS        0.3     1.5     1.95   0.0502 0.118       1.61 
# 4 Linear     0.3     1.5     1.92   0.0848 0.149       1.62 
# 5 RF_CF      0.3     1.5     2.16   0.157  0.194       1.78 
# 6 XGB_CF     0.3     1.5     2.17   0.171  0.202       1.72 
# 7 XGB_CF     1.5     1.5     2.02   0.0153 0.0957      0.289
# 8 RCS        1.5     1.5     1.94   0.0567 0.121       0.253
# 9 RF_CF      1.5     1.5     1.92   0.0753 0.122       0.328
# 10 Oracle     1.5     1.5     1.92   0.0847 0.123       0.231
# 11 GAM_CF     1.5     1.5     1.93   0.0691 0.124       0.260
# 12 Linear     1.5     1.5     1.90   0.0981 0.158       0.267
cat("开始运行包含 Oracle 和 XGBoost 的测试...\n")
test_res <- run_full_simulation(n_sim = 200, n = 1000, scenario = "1_NonLinear", calc_se = FALSE)
print(test_res)
# # A tibble: 12 × 7
# Method gamma_z gamma_w Mean_Est Abs_Bias   RMSE Mean_MSE_S
# <chr>    <dbl>   <dbl>    <dbl>    <dbl>  <dbl>      <dbl>
#   1 GAM_CF     0.3     1.5     2.01 0.0126   0.0826      1.63 
# 2 RCS        0.3     1.5     2.00 0.000870 0.0988      1.59 
# 3 Oracle     0.3     1.5     1.95 0.0522   0.105       1.60 
# 4 Linear     0.3     1.5     1.88 0.121    0.181       1.64 
# 5 RF_CF      0.3     1.5     2.20 0.204    0.227       1.78 
# 6 XGB_CF     0.3     1.5     2.21 0.213    0.233       1.72 
# 7 GAM_CF     1.5     1.5     1.99 0.00604  0.0885      0.231
# 8 RF_CF      1.5     1.5     1.96 0.0353   0.0946      0.326
# 9 RCS        1.5     1.5     2.01 0.00639  0.0959      0.226
# 10 XGB_CF     1.5     1.5     2.05 0.0531   0.0989      0.278
# 11 Oracle     1.5     1.5     1.86 0.141    0.166       0.229
# 12 Linear     1.5     1.5     1.87 0.132    0.192       0.283
cat("开始运行包含 Oracle 和 XGBoost 的测试...\n")
test_res <- run_full_simulation(n_sim = 200, n = 1000, scenario = "0_Linear", calc_se = FALSE)
print(test_res)
# # A tibble: 12 × 7
# Method gamma_z gamma_w Mean_Est Abs_Bias   RMSE Mean_MSE_S
# <chr>    <dbl>   <dbl>    <dbl>    <dbl>  <dbl>      <dbl>
#   1 GAM_CF     0.3     1.5     2.01 0.0145   0.0873      1.62 
# 2 Oracle     0.3     1.5     2.00 0.00428  0.0896      1.59 
# 3 Linear     0.3     1.5     2.01 0.00823  0.0953      1.59 
# 4 RCS        0.3     1.5     2.00 0.00226  0.0967      1.59 
# 5 RF_CF      0.3     1.5     2.21 0.209    0.228       1.77 
# 6 XGB_CF     0.3     1.5     2.22 0.216    0.238       1.71 
# 7 Oracle     1.5     1.5     2.00 0.000395 0.0848      0.225
# 8 Linear     1.5     1.5     2.00 0.000713 0.0850      0.224
# 9 GAM_CF     1.5     1.5     2.01 0.0104   0.0858      0.229
# 10 RF_CF      1.5     1.5     1.98 0.0196   0.0859      0.311
# 11 RCS        1.5     1.5     2.01 0.00583  0.0911      0.224
# 12 XGB_CF     1.5     1.5     2.06 0.0565   0.0999      0.265