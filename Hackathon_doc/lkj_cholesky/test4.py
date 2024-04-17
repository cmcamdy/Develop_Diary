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

# 示例
p = paddle.to_tensor([[-0.23046529], [-0.30970311], [0.23422813], [-0.12936383], [-0.40200305], [-0.36107415]])
matrix = construct_matrix_lower(p)
print(matrix)
