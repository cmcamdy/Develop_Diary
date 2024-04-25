import paddle

# 定义一个多维向量值函数
def func(x):
    return paddle.sum(x * x, axis=1)

# 示例输入，一个 2x2 的多维 Tensor
x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], stop_gradient=False)

# 计算雅可比矩阵
y = func(x)
print(y.shape)
jacobian = []
for i in range(y.shape[0]):
    grads = paddle.grad(
        outputs=y[i],
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    jacobian.append(grads)
jacobian = paddle.stack(jacobian, axis=0)

print(jacobian.numpy())
