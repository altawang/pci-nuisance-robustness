library(dplyr)
library(parallel)
library(mgcv)
library(splines)
library(ranger)
library(xgboost)
# n = 500, 1000, 2000
# ------------------------------------------------------------------------------
# 1. DGP（保持不变）
# ------------------------------------------------------------------------------
generate_nonlinear_X_data <- function(n_sample, scenario = "1_NonLinear",
                                      gamma_z = 1, gamma_w = 1,beta_U = 1) {
  X1 <- rnorm(n_sample)
  X2 <- rnorm(n_sample)
  X3 <- rbinom(n_sample, 1, 0.5)
  U  <- rnorm(n_sample)
  beta_true <- 2.0
  Z <- gamma_z * U + 0.1*X1 + 0.1*X2 - 0.1*X3 + rnorm(n_sample, 0, 0.5)
  
  if (scenario == "0_Linear") {
    prob_A  <- plogis(-1.5 + 0.5*U + 0.4*X1 - 0.2*X2 + 0.5*X3)
    W_clean <- gamma_w*U + 0.2*X1 + 0.2*X2 - 0.2*X3
  } else if (scenario == "1_NonLinear") {
    prob_A  <- plogis(-1.5 + 0.5*U + 0.8*sin(1.5*X1) - 0.2*X2 + 0.5*X3)
    W_clean <- gamma_w*U + 0.5*sin(1.5*X1) + 0.2*X2 - 0.2*X3
  } else {
    prob_A  <- plogis(-1.5 + 0.5*U + 0.8*sin(1.5*X1)*X3 - 0.2*X2 + 0.5*X3)
    W_clean <- gamma_w*U + 0.5*sin(1.5*X1)*X3 + 0.2*X2 - 0.2*X3
  }
  
  A <- rbinom(n_sample, 1, prob_A)
  
  if (scenario == "0_Linear") {
    Y_mean <- beta_true*A + beta_U*U + 0.8*X1 + 0.5*X2 + 0.5*X3
  } else if (scenario == "1_NonLinear") {
    Y_mean <- beta_true*A + beta_U*U + 1.5*cos(1.5*X1) + 0.5*X2 + 0.5*X3
  } else {
    Y_mean <- beta_true*A + beta_U*U + 1.5*cos(1.5*X1)*X3 + 0.5*X2 + 0.5*X3
  }
  
  W <- W_clean + rnorm(n_sample, 0, 0.5)
  Y <- Y_mean + rnorm(n_sample, 0, 1)
  
  data.frame(Y, A, Z, W, X1, X2, X3, W_true = W_clean)
}

# ------------------------------------------------------------------------------
# 2. 通用 Cross-Fitting 框架（统一）
# ------------------------------------------------------------------------------
estimate_S_hat <- function(dat, method, K = 5) {
  n <- nrow(dat)
  
  # 固定 fold（保证可重复性）
  folds <- sample(rep(1:K, length.out = n))
  S_hat <- numeric(n)
  
  for (k in 1:K) {
    
    tr  <- dat[folds != k, ]
    val <- dat[folds == k, ]
    
    # -----------------------------
    # 1. 拟合模型
    # -----------------------------
    if (method == "Linear") {
      fit <- lm(W ~ A + Z + X1 + X2 + X3, data = tr)
    }
    
    if (method == "Spline") {
      fit <- lm(W ~ A + Z + ns(X1, df = 6) * X3 + X2, data = tr)
    }
    
    if (method == "GAM") {
      fit <- mgcv::gam(
        W ~ A + Z + s(X1, by = factor(X3)) + X2 + factor(X3),
        data = tr
      )
    }
    
    if (method == "RF") {
      fit <- ranger::ranger(
        W ~ A + Z + X1 + X2 + X3,
        data = tr,
        num.trees = 500,       # 🔴 防止不稳定
        min.node.size = 5      # 🔴 控制过拟合
      )
    }
    
    if (method == "XGB") {
      
      X_tr  <- model.matrix(W ~ A + Z + X1 + X2 + X3, data = tr)[, -1]
      X_val <- model.matrix(W ~ A + Z + X1 + X2 + X3, data = val)[, -1]
      y_tr  <- tr$W
      
      fit <- xgboost::xgboost(
        x = X_tr,
        y = y_tr,
        nrounds = 200,
        objective = "reg:squarederror",
        max_depth = 3,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        verbose = 0
      )
    }
    
    # -----------------------------
    # 2. 预测（统一写法）
    # -----------------------------
    if (method == "RF") {
      pred <- predict(fit, data = val)$predictions
    } else if (method == "XGB") {
      pred <- predict(fit, newdata = X_val)
    } else {
      pred <- predict(fit, newdata = val)
    }
    
    # -----------------------------
    # 3. 防数值爆炸（非常关键）
    # -----------------------------
    pred[is.na(pred)] <- mean(tr$W)
    
    S_hat[folds == k] <- pred
  }
  
  return(S_hat)
}
# ------------------------------------------------------------------------------
# 3. 统一第二阶段（关键修正）
# ------------------------------------------------------------------------------
estimate_beta <- function(dat, S_hat) {
  dat$S_hat <- S_hat
  
  fit <- lm(Y ~ A + S_hat + ns(X1, df=6)*X3 + X2, data = dat)
  
  c(est = coef(fit)["A"],
    se  = summary(fit)$coef["A", "Std. Error"])
}

# ------------------------------------------------------------------------------
# 4. 单次模拟
# ------------------------------------------------------------------------------
run_one <- function(n, scenario, method, gamma_z = 1, gamma_w = 1) {
  
  # -----------------------------
  # 1. 生成数据
  # -----------------------------
  dat <- generate_nonlinear_X_data(
    n_sample = n,
    scenario = scenario,
    gamma_z = gamma_z,
    gamma_w = gamma_w
  )
  
  # -----------------------------
  # 2. 第一阶段（cross-fitting）
  # -----------------------------
  S_hat <- estimate_S_hat(dat, method)
  
  # 防止异常值污染（关键）
  S_hat[is.na(S_hat)] <- mean(dat$W)
  
  # -----------------------------
  # 3. 第二阶段
  # -----------------------------
  beta <- estimate_beta(dat, S_hat)
  
  # -----------------------------
  # 4. nuisance 误差（核心指标）
  # -----------------------------
  mse_S <- mean((S_hat - dat$W_true)^2)
  
  # -----------------------------
  # 5. 线性 F-stat（IV-style proxy strength）
  # -----------------------------
  fit_full <- lm(W ~ Z + A + X1 + X2 + X3, data = dat)
  fit_red  <- lm(W ~ A + X1 + X2 + X3, data = dat)
  
  # 用 anova 更稳健（避免手算误差）
  anova_res <- anova(fit_red, fit_full)
  
  f_stat <- as.numeric(anova_res$F[2])
  
  # 防止数值异常
  if (is.na(f_stat) || f_stat < 0) f_stat <- 0
  
  # -----------------------------
  # 6. ML-based proxy strength（新增亮点）
  # -----------------------------
  # 类似 first-stage R²（但更 general）
  R2_proxy <- suppressWarnings(cor(S_hat, dat$W)^2)
  if (is.na(R2_proxy)) R2_proxy <- 0
  
  # -----------------------------
  # 7. 输出
  # -----------------------------
  data.frame(
    Method = method,
    Est = beta[1],
    SE = beta[2],
    
    # 核心性能指标
    MSE_S = mse_S,
    
    # 两种 proxy strength
    F_stat = f_stat,        # 线性（IV-style）
    R2_proxy = R2_proxy,    # ML-based
    
    # DGP 参数
    gamma_z = gamma_z,
    gamma_w = gamma_w
  )
}
# ------------------------------------------------------------------------------
# 5. 主模拟函数
# ------------------------------------------------------------------------------
run_sim <- function(n_sim = 300, n = 1000, scenario = "1_NonLinear") {
  
  methods <- c("Linear","Spline","GAM","RF","XGB")
  
  # 正确的 grid（显式 data.frame）
  gamma_grid <- expand.grid(
    gamma_z = c(0.2, 0.5, 1),
    gamma_w = c(0.2, 0.5, 1)
  )
  
  # -----------------------------
  # 主循环（调用 run_one）
  # -----------------------------
  results <- mclapply(1:n_sim, function(i) {
    
    bind_rows(
      lapply(1:nrow(gamma_grid), function(j) {
        
        g <- gamma_grid[j, ]
        
        bind_rows(
          lapply(methods, function(m) {
            run_one(
              n = n,
              scenario = scenario,
              method = m,
              gamma_z = g$gamma_z,
              gamma_w = g$gamma_w
            )
          })
        )
        
      })
    )
    
  }, mc.cores = 1)
  
  beta_true <- 2.0
  
  # -----------------------------
  # 汇总（论文表格）
  # -----------------------------
  summary_res <- bind_rows(results) %>%
    mutate(
      Bias = Est - beta_true,
      Covered = (Est - 1.96*SE <= beta_true & Est + 1.96*SE >= beta_true)
    ) %>%
    group_by(Method, gamma_z, gamma_w) %>%
    summarise(
      Bias = mean(Bias),
      Emp_SD = sd(Est),
      Avg_SE = mean(SE),
      Coverage = mean(Covered),
      RMSE = sqrt(mean(Bias^2)),
      
      # nuisance quality
      MSE_S = mean(MSE_S),
      
      # proxy strength
      Avg_F = mean(F_stat),
      Avg_R2 = mean(R2_proxy),
      
      .groups = "drop"
    )
  
  return(summary_res)
}
# ------------------------------------------------------------------------------
# 6. 运行
# ------------------------------------------------------------------------------
set.seed(2026)
res <- run_sim(n_sample, scenario = "1_NonLinear",
               gamma_z = 1, gamma_w = 1,beta_U = 1)
print(res)