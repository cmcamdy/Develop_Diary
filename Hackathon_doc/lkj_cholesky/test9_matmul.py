import paddle 




x = paddle.to_tensor(
    [[ 1.        ,  0.24601485, -0.14606369],
        [1.        ,  0.24601485, -0.14606369],
        [ 1.        ,  0.24601485, -0.14606369]],
        place=paddle.CPUPlace()
)

print(x)
print(x @ x.T)


