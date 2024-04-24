为输入创建一个可微分的张量
import torch
# x = torch.randn(3, requires_grad=True)
x = torch.tensor([0.24601485, -0.14606369, -0.49986067])

# 计算雅可比矩阵
J = torch.autograd.functional.jacobian(_tril_cholesky_to_tril_corr, x)

# 计算行列式的符号和对数绝对值
# sign, logabsdet = torch.linalg.slogdet(J)
logabsdet = torch.linalg.slogdet(J).logabsdet

print("Log absolute determinant:", logabsdet.item())
