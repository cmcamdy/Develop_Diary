import paddle
import torch

# 设置 PaddlePaddle 的随机种子
paddle.seed(42)
# 生成一个形状为 [3, 1] 的标准正态分布随机数张量
paddle_tensor = paddle.randn([3, 1])
print("PaddlePaddle random tensor:\n", paddle_tensor.numpy())

# 设置 PyTorch 的随机种子
torch.manual_seed(42)
# 生成一个形状为 [3, 1] 的标准正态分布随机数张量
torch_tensor = torch.randn([3, 1])
print("PyTorch random tensor:\n", torch_tensor.numpy())
