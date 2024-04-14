import torch
from torch.distributions import MultivariateNormal

# 均值向量 (mean vector)
mean = torch.tensor([0.0, 0.0])

# 协方差矩阵 (covariance matrix)
covariance = torch.tensor([[1.0, 0.0],
                           [0.0, 1.0]])

# 创建一个多元正态分布对象
multivariate_normal = MultivariateNormal(mean, covariance)

# 从分布中采样
sample = multivariate_normal.sample()
print("Sample from the distribution:", sample)

# 计算给定样本的概率密度
log_prob = multivariate_normal.log_prob(sample)
print("Log probability density of the sample:", log_prob)
