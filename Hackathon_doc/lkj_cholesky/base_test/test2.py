import paddle
import torch
import numpy as np
import random

# 设置 PaddlePaddle 的随机数种子
paddle.seed(42)
random.seed(42)
np.random.seed(42)

# 设置 PyTorch 的随机数种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # 如果使用多个GPU
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Beta 分布的参数
alpha = 2.0
beta = 5.0

# 在 PaddlePaddle 中采样
paddle_beta = paddle.distribution.Beta(alpha, beta)
paddle_samples = paddle_beta.sample([10]).numpy()
print("PaddlePaddle Beta samples:")
print(paddle_samples)

# 在 PyTorch 中采样
torch_beta = torch.distributions.beta.Beta(alpha, beta)
torch_samples = torch_beta.sample((10,)).numpy()
print("PyTorch Beta samples:")
print(torch_samples)
