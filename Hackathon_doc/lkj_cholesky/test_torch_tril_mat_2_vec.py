import torch  
  
def tril_matrix_to_vec(mat, diag: int = 0):  
    n = mat.shape[-1]  
      
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):  
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")  
  
    arange = torch.arange(n, device=mat.device)  
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)  
    vec = mat[..., tril_mask]  
    return vec

def vec_to_tril_matrix(vec: torch.Tensor, diag: int = 0) -> torch.Tensor:
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    n = round(n.item()) if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat




def tril_cholesky_to_tril_corr(x):
    x = vec_to_tril_matrix(x, -1)
    diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
    x = x + diag
    return tril_matrix_to_vec(x @ x.T, -1)  

# 创建一个3x3的下三角矩阵  
# matrix = torch.tensor([[[1, 0, 0], [2, 3, 0], [4, 5, 6]],
#                         [[1, 0, 0], [2, 3, 0], [4, 5, 6]]], dtype=torch.float32)  
matrix = torch.tensor([[1, 0, 0], [2, 3, 0], [4, 5, 6]], dtype=torch.float32)  
  
# 调用函数将矩阵转换为向量  
# vector = tril_matrix_to_vec(matrix, diag=-1)  

vector = torch.tensor([ 0.65644526,  0.43036091, -0.41606992], dtype=torch.float32)
# 输出结果  
print("Original Matrix:")  
print(matrix)  
print("\nConverted Vector:")  
print(vector)

result = tril_cholesky_to_tril_corr(vector)
print(result)