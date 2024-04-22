import paddle  
  
# 原始tensor  
original_tensor = paddle.to_tensor([[1, 0], [2, 0], [2, 1]], dtype='int64')  
  
# 创建索引矩阵  
index_matrix = paddle.arange(0, 2).unsqueeze(1).tile([3, 1])  # 创建[0, 0, 1, 1, 2, 2]的索引  

print(index_matrix)
# # 重复原始tensor的行  
repeated_tensor = paddle.repeat_interleave(original_tensor, 2, axis=0)  
  
# # 合并索引和重复的tensor  
result_tensor = paddle.concat([index_matrix, repeated_tensor], axis=1)  
  
  
print(result_tensor)