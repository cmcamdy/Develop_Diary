# 为输入创建一个可微分的张量
import torch

def func(x):
    return x
x = torch.tensor([-0.91035604,  0.55142891, -0.77994263])

# 计算雅可比矩阵
J = torch.autograd.functional.jacobian(func, x)
print("Log absolute determinant:", J)


# 计算行列式的符号和对数绝对值
# sign, logabsdet = torch.linalg.slogdet(J)
# logabsdet = torch.linalg.slogdet(J).logabsdet

# print("Log absolute determinant:", logabsdet.item())
