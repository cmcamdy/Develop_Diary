import torch
import matplotlib.pyplot as plt

def sample_student_t(df, loc, scale, num_samples):
    distribution = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
    samples = distribution.sample((num_samples,))
    return samples

def probability_density_student_t(samples, df, loc, scale):
    distribution = torch.distributions.StudentT(df=df, loc=loc, scale=scale)
    log_prob = distribution.log_prob(samples)
    return log_prob.exp()

# 定义自由度、均值和尺度参数
df = 5.0
loc = 0.0
scale = 1.0

# 从学生 t 分布中采样样本
num_samples = 1000
samples = sample_student_t(df, loc, scale, num_samples)

# 计算样本的概率密度
pdf_values = probability_density_student_t(samples, df, loc, scale)

# 绘制直方图和概率密度函数
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, density=True, alpha=0.6, label='Samples')
plt.plot(samples, pdf_values, 'r.', label='PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Student t Distribution (df={}, loc={}, scale={})'.format(df, loc, scale))
plt.legend()

# 保存图像到文件
plt.savefig('student_t_distribution.png', dpi=300, bbox_inches='tight')

# 显示图像
# plt.show()
