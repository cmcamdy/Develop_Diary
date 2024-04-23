import paddle
import math

def vec_to_tril_matrix(p):
    """
    Constructs a batch of lower triangular matrices from a given input tensor `p`.
    """
    # p.shape = [other_dims, L, 1]
    # Calculate the dimension of the square matrix based on the last but one dimension of `p`
    dim = int((math.sqrt(paddle.to_tensor(1 + 8*p.shape[-2])) + 1) / 2)
    
    # Flatten the input tensor to a 1D array 
    p_flatten = p.flatten()

    # Define the output shape, which adds two dimensions for the square matrix
    output_shape = tuple(p.shape[:-2]) + (dim, dim)
    shape0 = p_flatten.shape[0] // p.shape[-2]

    # Create index_matrix = [index0, rows, cols]
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    mask = rows > cols
    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    repeated_lower_indices = paddle.repeat_interleave(lower_indices, shape0, axis=0)  
    index0 = paddle.arange(shape0).unsqueeze(1).tile([dim, 1]) 
    index_matrix = paddle.concat([index0, repeated_lower_indices], axis=1)  
    
    # Sort the indices
    sorted_indices = paddle.argsort(index_matrix[:, 0])
    index_matrix = index_matrix[sorted_indices]
    
    # Set the value
    matrix = paddle.zeros(shape=(shape0, dim, dim), dtype=p.dtype)
    matrix = paddle.scatter_nd_add(matrix, index_matrix, p_flatten).reshape(output_shape)
    
    return matrix

# 示例使用
# p = paddle.to_tensor([[-0.23046529], [-0.30970311], [0.23422813]])
p = paddle.to_tensor([[[[-0.23046529], [-0.30970311], [0.23422813]],[[-0.12936383], [-0.40200305], [-0.36107415]]], [[[-0.23046529], [-0.30970311], [0.23422813]],[[-0.12936383], [-0.40200305], [-0.36107415]]]])
lower_tri_matrices = vec_to_tril_matrix(p)
# print("Shape of output:", lower_tri_matrices.shape)
