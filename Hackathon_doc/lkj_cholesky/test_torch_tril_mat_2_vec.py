import torch  
  
def tril_matrix_to_vec(mat, diag: int = 0):  
    n = mat.shape[-1]  
      
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):  
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")  
  
    arange = torch.arange(n, device=mat.device)  
    import pdb; pdb.set_trace()
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)  
    vec = mat[..., tril_mask]  
    import pdb; pdb.set_trace()
      
    return vec


# 创建一个3x3的下三角矩阵  
matrix = torch.tensor([[[1, 0, 0],   
                       [2, 3, 0],   
                       [4, 5, 6]],
                       [[1, 0, 0],   
                       [2, 3, 0],   
                       [4, 5, 6]]], dtype=torch.float32)  
  
# 调用函数将矩阵转换为向量  
vector = tril_matrix_to_vec(matrix, diag=-1)  
  
# 输出结果  
print("Original Matrix:")  
print(matrix)  
print("\nConverted Vector:")  
print(vector)