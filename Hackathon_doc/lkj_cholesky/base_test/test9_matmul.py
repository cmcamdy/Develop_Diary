import paddle 




x = paddle.to_tensor(
        [[[ 1.        ,  0.24601485, -0.14606369],
          [1.        ,  0.24601485, -0.14606369],
          [ 1.        ,  0.24601485, -0.14606369]],
         [[ 1.        ,  0.24601485, -0.14606369],
          [1.        ,  0.24601485, -0.14606369],
          [ 1.        ,  0.24601485, -0.14606369]]]
)

print(x.shape)
print(x @ x.T)


