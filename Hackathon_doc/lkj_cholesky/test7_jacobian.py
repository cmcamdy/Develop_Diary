import paddle
import math
paddle.set_device('cpu')

# import torch
def vec_to_tril_matrix(p, diag=0):
    """
    Constructs a batch of lower triangular matrices from a given input tensor `p`.
    """
    # p.shape = [other_dims, L, 1]
    # Calculate the dimension of the square matrix based on the last but one dimension of `p`
    last_dim = p.shape[-2]
    
    dim = int(math.sqrt(paddle.to_tensor(1 + 8 * last_dim))/ 2 - diag)
    
    # Flatten the input tensor to a 1D array 
    p_flatten = p.flatten()

    # Define the output shape, which adds two dimensions for the square matrix
    output_shape = tuple(p.shape[:-2]) + (dim, dim)
    shape0 = p_flatten.shape[0] // last_dim

    # Create index_matrix = [index0, rows, cols]
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    mask = rows > cols
    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    repeated_lower_indices = paddle.repeat_interleave(lower_indices, shape0, axis=0)  
    index0 = paddle.arange(shape0).unsqueeze(1).tile([last_dim, 1]) 
    index_matrix = paddle.concat([index0, repeated_lower_indices], axis=1)  
    
    # Sort the indices
    sorted_indices = paddle.argsort(index_matrix[:, 0])
    index_matrix = index_matrix[sorted_indices]
    
    # Set the value
    matrix = paddle.zeros(shape=(shape0, dim, dim), dtype=p.dtype)
    matrix = paddle.scatter_nd_add(matrix, index_matrix, p_flatten).reshape(output_shape)
    
    return matrix

def tril_matrix_to_vec(mat: paddle.Tensor, diag: int = 0) -> paddle.Tensor:  
    r"""  
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector  
    which comprises of lower triangular elements from the matrix in row order.  
    """  
    out_shape = mat.shape[:-2]
    n = mat.shape[-1]  
    if diag < -n or diag >= n:  
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")  
  
    rows, cols = paddle.meshgrid(paddle.arange(n), paddle.arange(n))
    tril_mask = diag + rows >= cols 
    
    vec_len = (n + diag) * (n + diag + 1)// 2
    out_shape.append(vec_len)
    
    # Use the mask to index the lower triangular elements from the input matrix  
    vec = paddle.masked_select(mat, tril_mask).reshape(out_shape)
    # vec = paddle.masked_select(mat, tril_mask)
    return vec  
  
# 定义一个简单的函数作为示例
def _tril_cholesky_to_tril_corr(x):
    # import pdb; pdb.set_trace()
    print(x.shape)
    x = x.unsqueeze(-1)
    print(x.shape)
    x = vec_to_tril_matrix(x, -1)
    print(x)
    diag = (1 - (x * x).sum(-1)).sqrt().diag_embed().cpu()
    print("diag:",diag)
    x = x + diag
    print("x:",x)
    print("xT:",x.T)
    print("x @ x.T:", paddle.matmul(x, x, transpose_y=True))
    # import pdb; pdb.set_trace()
    return tril_matrix_to_vec(paddle.matmul(x, x, transpose_y=True), -1)

# 为输入创建一个可微分的张量
# import torch
# # x = torch.randn(3, requires_grad=True)
# x = torch.tensor([0.24601485, -0.14606369, -0.49986067])

# # 计算雅可比矩阵
# J = torch.autograd.functional.jacobian(_tril_cholesky_to_tril_corr, x)

# # 计算行列式的符号和对数绝对值
# # sign, logabsdet = torch.linalg.slogdet(J)
# logabsdet = torch.linalg.slogdet(J).logabsdet

# print("Log absolute determinant:", logabsdet.item())



# 为输入创建一个可微分的张量
# x = paddle.to_tensor([0.24601485, -0.14606369, -0.49986067], stop_gradient=False)
x = paddle.to_tensor( [[-0.30670628,  0.17507216, -0.44365808], 
                       [ 0.07543492, -0.35833275,  0.10217478]], stop_gradient=False)

print(x)
# 手动计算雅可比矩阵（这里使用一个简化的方法，仅适用于特定函数）
def compute_jacobian(x, func):
    jacobian_matrix = []
    outputs = func(x)
    for i in range(len(x)):
        x = x.cpu()
        # 计算每个输出相对于整个输入向量的梯度
        
        grad = paddle.grad(outputs=outputs[i], inputs=x, create_graph=False)[0]
        jacobian_matrix.append(grad)
    return paddle.stack(jacobian_matrix, axis=0)

J = compute_jacobian(x, _tril_cholesky_to_tril_corr)

# 计算行列式的符号和对数绝对值
# sign, logabsdet = paddle.linalg.slogdet(J)
_, logabsdet = paddle.linalg.slogdet(J)

print("Log absolute determinant:", logabsdet.numpy())