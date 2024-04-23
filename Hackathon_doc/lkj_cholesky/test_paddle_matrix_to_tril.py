import numpy as np  
import paddle  
 
  
def tril_indices(n, k=0):
    """
    Returns the indices of the lower triangular part of an n x n matrix, including the k-th diagonal.
    """
    full_matrix = paddle.ones((n, n), dtype='int32')
    tril_matrix = paddle.tril(full_matrix, diagonal=k)
    rows, cols = paddle.nonzero(tril_matrix, as_tuple=True)
    return rows.flatten(), cols.flatten()

def matrix_to_tril(x, diagonal=0):  
    """  
    Extracts the lower triangular part of a matrix or a batch of matrices `x`,   
    including the specified diagonal.  
    """  
    matrix_dim = x.shape[-1]  
      
    rows, cols = tril_indices(matrix_dim, diagonal)  
      
    return x[..., rows, cols]


  
# 假设上面提供的tril_indices和matrix_to_tril函数已经定义  
  
# 创建一个4x4的矩阵  
matrix = paddle.to_tensor([[1, 2, 3, 4],  
                           [5, 6, 7, 8],  
                           [9, 10, 11, 12],  
                           [13, 14, 15, 16]], dtype='float32')  
  
print("原始矩阵:")  
print(matrix.numpy())  
  
# 使用matrix_to_tril函数提取下三角部分  
tril_matrix_elements = matrix_to_tril(matrix)  
  
# 由于tril_matrix_elements是一维的，我们可以将其重新排列成二维的下三角矩阵形式以便查看  
tril_indices = tril_indices(4)  # 获取4x4矩阵的下三角索引  
tril_matrix = paddle.zeros_like(matrix)  # 创建一个与原始矩阵形状相同的零矩阵  
tril_matrix[..., tril_indices[0], tril_indices[1]] = tril_matrix_elements  # 填充下三角数据  
  
print("\n下三角矩阵:")  
print(tril_matrix.numpy())