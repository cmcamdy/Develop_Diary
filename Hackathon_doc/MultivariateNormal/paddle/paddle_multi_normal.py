import paddle
import numpy as np

# 设置 PaddlePaddle 运行模式
paddle.set_device('cpu')

def sample_multivariate_normal(mean, cov, num_samples):
    distribution = paddle.distribution.MultivariateNormal(loc=mean, scale_tril=cov)
    samples = distribution.sample([num_samples])
    return samples.numpy()

def probability_density_multivariate_normal(samples, mean, cov):
    distribution = paddle.distribution.MultivariateNormal(loc=mean, scale_tril=cov)
    log_prob = distribution.log_prob(paddle.to_tensor(samples))
    return np.exp(log_prob.numpy())

# 定义均值向量和协方差矩阵
mean = paddle.to_tensor([0.0, 0.0])
cov = paddle.to_tensor([[1.0, 0.0], [0.0, 1.0]])

# 从多元正态分布中采样样本
num_samples = 1000
samples = sample_multivariate_normal(mean, cov, num_samples)

# 计算样本的概率密度
pdf_values = probability_density_multivariate_normal(samples, mean, cov)

print("Samples:\n", samples)
print("Probability Density Values:\n", pdf_values)
