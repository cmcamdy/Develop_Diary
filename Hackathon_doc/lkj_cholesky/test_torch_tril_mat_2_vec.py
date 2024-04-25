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


vector = torch.tensor([-0.91035604,  0.55142891, -0.67164946], dtype=torch.float32)
# vector = torch.tensor([[ 0.48528507,  0.40442318,  0.61717522],
#                         [-0.41349775,  0.64750606, -0.46594250]], dtype=torch.float32)
# 输出结果  
print("\nConverted Vector:")  
print(vector)

result = tril_cholesky_to_tril_corr(vector)
print(result)