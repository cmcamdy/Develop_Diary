import paddle
import math

def construct_matrix_lower(p):
    
    # cal dim
    dim = int((math.sqrt(1 + 8*p.shape[0]) + 1) / 2)
    
    # 创建一个dim x dim的零矩阵
    matrix = paddle.zeros(shape=[dim, dim], dtype='float32')
    import pdb; pdb.set_trace()
    # 计算下三角矩阵的索引
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    mask = rows > cols
    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    
    print(lower_indices)
    # 将p中的元素填充到matrix的相应位置
    matrix = paddle.scatter_nd_add(matrix, lower_indices, paddle.flatten(p))
    
    return matrix

def construct_matrix_lower_n_dim(p):
    """
    Constructs lower triangular matrices from an N-D tensor `p` containing the elements of the lower triangular parts.
    
    Args:
        p (paddle.Tensor): An N-D tensor of shape (..., h, 1) where `h` is the number of elements in the lower triangular part,
                           including the diagonal. The length of `h` must be a triangular number (i.e., 1, 3, 6, 10, ...).
    
    Returns:
        paddle.Tensor: An N-D tensor of shape (..., dim, dim) where `dim` is the side length of the square lower triangular
                       matrices, filled from the input tensor `p`.
    """
    if p.shape[-1] != 1:
        raise ValueError("The last dimension of the input tensor must be 1.")
    
    # Flatten the last two dimensions
    p_flat = paddle.flatten(p, start_axis=-2)
    
    # Calculate the dimension of the square matrix
    h = p_flat.shape[-1]
    dim = int((math.sqrt(1 + 8 * h) + 1) / 2)
    
    print(dim, h)
    # Create an output tensor of zeros
    output_shape = tuple(p.shape[:-2]) + (dim, dim)
    matrix = paddle.zeros(output_shape, dtype=p.dtype)
    
    # Generate indices for the lower triangular part
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim), indexing='ij')
    mask = rows >= cols
    # # Iterate over the first N-2 dimensions
    indices = paddle.where(mask)
    for index in paddle.ndindex(*p.shape[:-2]):
        # Extract the elements for the current lower triangular matrix
        elements = p_flat[index].reshape([-1])
        # Fill the lower triangular part of the matrix
        matrix[index][indices[:, 0], indices[:, 1]] = elements
    
    return matrix

# Example usage
# p = paddle.randn([4, 10])  # Example input tensor, where 10 is a triangular number (4x4 matrix)
# lower_tri_matrices = construct_matrix_lower_n_dim(p)
# print("Shape of output:", lower_tri_matrices.shape)


# 示例
p = paddle.to_tensor([[-0.23046529], [-0.30970311], [0.23422813], [-0.12936383], [-0.40200305], [-0.36107415]])
# matrix = construct_matrix_lower(p)
matrix = construct_matrix_lower_n_dim(p)
print(matrix)
