# import paddle

# # 创建一个初始张量，维度为[2, 5, 5]
# input = paddle.arange(50, dtype='float32').reshape([2, 5, 5])

# # 索引张量，指明要更新的位置
# index = paddle.to_tensor([[0, 1, 1], [1, 3, 3]])

# # 更新值张量，与index中的索引对应
# updates = paddle.to_tensor([1.0, 2.0])

# # 执行scatter_nd_add
# result = paddle.scatter_nd_add(input, index, updates)

# print(input)
# print(result)

import paddle

# 创建一个二维张量
x = paddle.to_tensor([[3, 1, 2], 
                      [9, 6, 8], 
                      [5, 4, 7]], dtype='float32')

# 选择排序的列，例如第一列
column_index = 0

# 根据第一列的值对行索引进行排序
sorted_indices = paddle.argsort(x[:, column_index])

# 使用这些索引来重新排列整个张量
sorted_x = x[sorted_indices]

print("Original Tensor:\n", x.numpy())
print("Sorted Tensor by Column 0:\n", sorted_x.numpy())

