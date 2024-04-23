import torch
import paddle

# 定义一个简单的函数作为示例
def func(x):
    return x ** 2 + 3 * x

# 为输入创建一个可微分的张量
# x = torch.randn(3, requires_grad=True)
x = torch.tensor([ 0.24601485, -0.14606369, -0.49986067])

# 计算雅可比矩阵
J = torch.autograd.functional.jacobian(func, x)

# 计算行列式的符号和对数绝对值
# sign, logabsdet = torch.linalg.slogdet(J)
logabsdet = torch.linalg.slogdet(J).logabsdet

print("Log absolute determinant:", logabsdet.item())



# 为输入创建一个可微分的张量
x = paddle.to_tensor([ 0.24601485, -0.14606369, -0.49986067], stop_gradient=False)

# 手动计算雅可比矩阵（这里使用一个简化的方法，仅适用于特定函数）
def compute_jacobian(x, func):
    jacobian_matrix = []
    for i in range(len(x)):
        # 计算每个输出相对于整个输入向量的梯度
        grad = paddle.grad(outputs=func(x)[i], inputs=x, create_graph=False)[0]
        jacobian_matrix.append(grad)
    return paddle.stack(jacobian_matrix, axis=0)

J = compute_jacobian(x, func)

# 计算行列式的符号和对数绝对值
# sign, logabsdet = paddle.linalg.slogdet(J)
_, logabsdet = paddle.linalg.slogdet(J)

print("Log absolute determinant:", logabsdet.numpy())