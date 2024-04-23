import paddle  
  

def tril_matrix_to_vec(mat: paddle.Tensor, diag: int = 0) -> paddle.Tensor:  
    r"""  
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector  
    which comprises of lower triangular elements from the matrix in row order.  
    """  
    # import pdb; pdb.set_trace()
    out_shape = mat.shape[:-2]
    n = mat.shape[-1]  
    if diag < -n or diag >= n:  
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n-1}].")  
  
    rows, cols = paddle.meshgrid(paddle.arange(n), paddle.arange(n))
    tril_mask = diag + rows >= cols 
  
    # Use the mask to index the lower triangular elements from the input matrix  
    vec = paddle.masked_select(mat, tril_mask).reshape(out_shape)
    return vec  
  

# Example usage:  
mat = paddle.to_tensor([[[1, 0, 0], [2, 3, 0], [4, 5, 6]],
                        [[1, 0, 0], [2, 3, 0], [4, 5, 6]]], dtype='float32')  
result = tril_matrix_to_vec_v2(mat, diag=-1)  
print(result.numpy())