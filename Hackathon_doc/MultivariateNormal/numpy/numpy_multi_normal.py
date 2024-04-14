import numpy as np

# 均值向量 (mean vector)
mean = np.array([0.0, 0.0])

# 协方差矩阵 (covariance matrix)
covariance = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

# 从分布中采样
sample = np.random.multivariate_normal(mean, covariance)
print("Sample from the distribution:", sample)

# 计算给定样本的概率密度
from scipy.stats import multivariate_normal

# 创建一个多元正态分布对象
multivariate_normal_distribution = multivariate_normal(mean, covariance)

# 计算给定样本的概率密度
pdf = multivariate_normal_distribution.pdf(sample)
print("Probability density of the sample:", pdf)
