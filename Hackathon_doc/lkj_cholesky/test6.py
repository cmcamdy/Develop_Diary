import paddle  
import math

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
  
def tril_cholesky_to_tril_corr(x):
    x = x.unsqueeze(-1)
    x = vec_to_tril_matrix(x, -1)
    diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
    x = x + diag
    return tril_matrix_to_vec(x @ x.T, -1)

def tril_cholesky_to_tril_corr_paddle(x):
    x = paddle.unsqueeze(x, axis=-1)  # 调整unsqueeze的使用
    x = vec_to_tril_matrix(x, -1)  # 假设这个函数已经被适配为PaddlePaddle
    diag_elements = (1 - paddle.sum(x * x, axis=-1)).sqrt()
    diag = paddle.diag_embed(diag_elements)  # 构造对角矩阵
    x = x + diag
    return tril_matrix_to_vec(paddle.matmul(x, x.transpose([0, 2, 1])), -1)  # 假设这个函数已经被适配为PaddlePaddle

# Example usage:  
# mat = paddle.to_tensor([[[1, 0, 0], [2, 3, 0], [4, 5, 6]],
#                         [[1, 0, 0], [2, 3, 0], [4, 5, 6]]], dtype='float32') 
mat = paddle.to_tensor([[1, 0, 0], [2, 3, 0], [4, 5, 6]], dtype='float32')  
# result = tril_matrix_to_vec(mat, diag=-1)  

result = paddle.to_tensor([ 0.65644526,  0.43036091, -0.41606992], dtype='float32')  

x = tril_cholesky_to_tril_corr(result)
# x_pd = tril_cholesky_to_tril_corr_paddle(result)
# print(x, x_pd)
print(x)
