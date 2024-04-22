import paddle
import math

def construct_matrix_lower_n_dim(p):
    print(p.shape)
    dim = int((math.sqrt(paddle.to_tensor(1 + 8*p.shape[-2])) + 1) / 2)
    output_shape = tuple(p.shape[:-2]) + (dim, dim)
    # matrix = paddle.zeros(shape=[dim, dim], dtype=p.dtype)
    p_flatten = p.reshape((-1, p.shape[-2]))
    matrix = paddle.zeros(shape=(p_flatten.shape[0], dim, dim), dtype=p.dtype)
    
    print(p_flatten.shape)
    print(output_shape)
    
    rows, cols = paddle.meshgrid(paddle.arange(dim), paddle.arange(dim))
    # print("x, y, z:", x, y, z)

    mask = rows > cols
   
    lower_indices = paddle.stack([rows[mask], cols[mask]], axis=1)
    # repeated_lower_indices = paddle.repeat_interleave(lower_indices, p_flatten.shape[0], axis=0)  
    
    # index0 = paddle.arange(p_flatten.shape[0]).unsqueeze(1).tile([dim, 1]) 
    # index_matrix = paddle.concat([index0, repeated_lower_indices], axis=1)  
    
    # print(index0.shape)
    # # print(lower_indices)
    # print(repeated_lower_indices)
    # print(index_matrix)
    matrix = paddle.scatter_nd_add(matrix, lower_indices, p_flatten)
    
    print(matrix)
    return matrix


# 示例使用
# p = paddle.to_tensor([[-0.23046529], [-0.30970311], [0.23422813]])
p = paddle.to_tensor([[[[-0.23046529], [-0.30970311], [0.23422813]],[[-0.12936383], [-0.40200305], [-0.36107415]]],[[[-0.23046529], [-0.30970311], [0.23422813]],[[-0.12936383], [-0.40200305], [-0.36107415]]]])
lower_tri_matrices = construct_matrix_lower_n_dim(p)
# print("Shape of output:", lower_tri_matrices.shape)
