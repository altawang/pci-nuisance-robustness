# ==============================================================================
# 回归型近端因果推断 (PCI) 模拟研究脚本 (机制验证与严谨性升级版)
# ==============================================================================

library(dplyr)
library(parallel)
library(splines)   
library(sandwich)  
library(lmtest)    
library(ggplot2)   # 引入画图包用于机制验证

# ------------------------------------------------------------------------------
# 1. 数据生成机制 (严格控制 A 的生成为线性)
# ------------------------------------------------------------------------------
generate_master_data <- function(n = 1000, scenario = "1_NonLinear", 
                                 gamma_z = 1.5, gamma_w = 1.5) {
  X1 <- rnorm(n, 0, 1)
  X2 <- rnorm(n, 0, 1)
  X3 <- rbinom(n, 1, 0.5)
  U  <- rnorm(n, 0, 1)
  beta_true <- 2.0
  
  Z <- gamma_z * U + 0.1*X1 + 0.1*X2 - 0.1*X3 + rnorm(n, 0, 0.5)
  
  # 【修复 1】: 强制 A 的生成在所有场景下均为线性，彻底剥离 X->A 的错设干扰
  prob_A <- plogis(-1.5 + 0.5*U + 0.4*X1 - 0.2*X2 + 0.5*X3)
  A <- rbinom(n, 1, prob_A)
  
  if (scenario == "0_Linear") {
    W_clean <- gamma_w*U + 0.2*X1 + 0.2*X2 - 0.2*X3
    y_noA   <- 1.5*U + 0.8*X1 + 0.5*X2 + 0.5*X3
  } else if (scenario == "1_NonLinear") {
    # 非线性仅存在于 X->W 和 X->Y
    W_clean <- gamma_w*U + 0.5*sin(1.5*X1) + 0.2*X2 - 0.2*X3
    y_noA   <- 1.5*U + 2.0*cos(1.5*X1) + 0.5*X2 + 0.5*X3
  } 
  
  Y <- beta_true * A + y_noA + rnorm(n, 0, 1)
  W <- W_clean + rnorm(n, 0, 0.5)
  
  return(data.frame(Y=Y, A=A, Z=Z, W=W, X1=X1, X2=X2, X3=X3, W_clean=W_clean, U=U))
}

# ------------------------------------------------------------------------------
# 2. 估计器实现 (包含混合错设与敏感性分析)
# ------------------------------------------------------------------------------

est_oracle <- function(df, sc) {
  if (sc == "0_Linear") {
    f2 <- lm(Y ~ A + U + X1 + X2 + X3, data = df)
  } else if (sc == "1_NonLinear") {
    f2 <- lm(Y ~ A + U + cos(1.5*X1) + X2 + X3, data = df)
  }
  c(est = unname(coef(f2)["A"]), mse_s = 0)
}

# 【修复 2】: 拆解第一阶段与第二阶段的错设矩阵
est_hybrid <- function(df, model_s1 = "Linear", model_s2 = "Linear") {
  
  # 第一阶段: 预测 W
  if(model_s1 == "Linear") {
    f1 <- lm(W ~ A + Z + X1 + X2 + X3, data = df)
  } else if(model_s1 == "RCS") {
    f1 <- lm(W ~ A + Z + ns(X1, df=4) + ns(X2, df=4) + X3, data = df)
  }
  df$hatW <- predict(f1, newdata = df)
  mse_val <- mean((df$hatW - df$W_clean)^2)
  
  # 第二阶段: 提取因果效应 Y
  if(model_s2 == "Linear") {
    f2 <- lm(Y ~ A + hatW + X1 + X2 + X3, data = df)
  } else if(model_s2 == "RCS") {
    f2 <- lm(Y ~ A + hatW + ns(X1, df=4) + ns(X2, df=4) + X3, data = df)
  }
  
  c(est = unname(coef(f2)["A"]), mse_s = mse_val)
}

# 【修复 4】: Poly 多项式的敏感性分析 (支持动态传入 degree)
est_poly <- function(df, degree = 3) {
  f1 <- lm(W ~ A + Z + poly(X1, degree) + poly(X2, degree) + X3, data = df)
  df$hatW <- predict(f1, newdata = df)
  f2 <- lm(Y ~ A + hatW + poly(X1, degree) + poly(X2, degree) + X3, data = df)
  c(est = unname(coef(f2)["A"]), mse_s = mean((df$hatW - df$W_clean)^2))
}

# ------------------------------------------------------------------------------
# 3. 核心执行逻辑
# ------------------------------------------------------------------------------
run_one <- function(n, sc, method_name, gamma_z, gamma_w) {
  df <- generate_master_data(n, sc, gamma_z, gamma_w)
  
  res_tmp <- switch(method_name,
                    "Oracle"           = est_oracle(df, sc),
                    "Linear_Both"      = est_hybrid(df, "Linear", "Linear"),
                    "Linear_S1_RCS_S2" = est_hybrid(df, "Linear", "RCS"),
                    "RCS_S1_Linear_S2" = est_hybrid(df, "RCS", "Linear"),
                    "RCS_Both"         = est_hybrid(df, "RCS", "RCS"),
                    "Poly_d2"          = est_poly(df, degree = 2),
                    "Poly_d3"          = est_poly(df, degree = 3),
                    "Poly_d4"          = est_poly(df, degree = 4))
  
  return(data.frame(Method = method_name, Est = unname(res_tmp["est"]), 
                    MSE_S = unname(res_tmp["mse_s"]), 
                    gamma_z = gamma_z, gamma_w = gamma_w))
}

# 安全计算回归斜率和 P 值的辅助函数
safe_lm_slope <- function(y, x) { if(var(x) == 0) return(NA_real_) else return(coef(lm(y ~ x))[2]) }
safe_lm_pval  <- function(y, x) { if(var(x) == 0) return(NA_real_) else return(summary(lm(y ~ x))$coefficients[2,4]) }

run_full_simulation <- function(n_sim = 100, n = 1000, scenario = "1_NonLinear") {
  
  methods <- c("Oracle", "Linear_Both", "Linear_S1_RCS_S2", "RCS_S1_Linear_S2", "RCS_Both", 
               "Poly_d2", "Poly_d3", "Poly_d4")
  gamma_grid <- expand.grid(gamma_z = c(0.3, 1.5), gamma_w = c(1.5)) 
  
  results <- mclapply(1:n_sim, function(i) {
    set.seed(i + 2026)
    iter_res <- list()
    for(m in methods) {
      for(j in 1:nrow(gamma_grid)) {
        res <- run_one(n, scenario, m, gamma_grid$gamma_z[j], gamma_grid$gamma_w[j])
        iter_res[[length(iter_res)+1]] <- res
      }
    }
    bind_rows(iter_res)
  }, mc.cores = 1) 
  
  raw_data <- bind_rows(results) %>% mutate(Bias_sq = (Est - 2.0)^2)
  
  final_tab <- raw_data %>%
    group_by(Method, gamma_z, gamma_w) %>%
    summarise(
      Mean_Est = mean(Est),
      Abs_Bias = abs(mean(Est - 2.0)),
      RMSE = sqrt(mean(Bias_sq)),
      Mean_MSE_S = mean(MSE_S),
      # 【修复 3】: 引入线性回归的 Slope 和 P-value 进行机制验证
      Mech_Corr = ifelse(Method == "Oracle", NA_real_, cor(Bias_sq, MSE_S)),
      Mech_Slope= ifelse(Method == "Oracle", NA_real_, safe_lm_slope(Bias_sq, MSE_S)),
      Mech_Pval = ifelse(Method == "Oracle", NA_real_, safe_lm_pval(Bias_sq, MSE_S)),
      .groups = "drop"
    ) %>%
    arrange(gamma_z, RMSE)
  
  # 返回一个 List，包含汇总表和用于画散点图的原始数据
  return(list(Summary = final_tab, RawData = raw_data))
}

# ------------------------------------------------------------------------------
# 4. 运行验证与发表级机制作图 (ggplot2)
# ------------------------------------------------------------------------------
cat("运行核心机制测试 (100次模拟大约需 5 秒)...\n")
sim_output <- run_full_simulation(n_sim = 100, n = 1000, scenario = "1_NonLinear")

# 查看汇总表（包含 Slope 和 P-value）
print(sim_output$Summary)

# 【修复 3】：绘制机制验证散点图 (以 Linear_Both 和 RCS_Both 为例)
# 该图将直接揭示：第一阶段预测误差如何放大第二阶段的偏倚平方
plot_data <- sim_output$RawData %>% 
  filter(Method %in% c("Linear_Both", "RCS_Both"), gamma_z == 0.3) # 挑选弱代理下最能说明问题的两组

mech_plot <- ggplot(plot_data, aes(x = MSE_S, y = Bias_sq, color = Method)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE, formula = y ~ x) +
  labs(title = "误差传导机制验证 (Weak Proxy: γ_z = 0.3)",
       x = bquote("第一阶段预测误差 (" * MSE[S] * ")"),
       y = bquote("因果估计偏倚平方 (" * Bias^2 * ")")) +
  theme_minimal() +
  scale_color_manual(values = c("Linear_Both" = "#e74c3c", "RCS_Both" = "#2980b9"))

# print(mech_plot) # 运行此行即可看到精美的机制验证散点图