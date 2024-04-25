import paddle
import paddle.static as static

def matrix_to_tril_elements(x, diagonal=-1):
    """
    Extracts the elements of the lower triangular part of the input matrix or batch of matrices `x`,
    including the specified diagonal.
    """
    # 获取下三角矩阵的布尔掩码
    tril_mask = paddle.tril(paddle.ones_like(x), diagonal=diagonal)
    
    # 使用掩码获取下三角矩阵的元素
    tril_elements = paddle.masked_select(x, tril_mask.astype('bool'))
    return tril_elements

# 启用静态图模式
paddle.enable_static()

# 创建静态图程序
main_program = static.Program()
startup_program = static.Program()

with static.program_guard(main_program, startup_program):
    # 创建输入的占位符
    dim = 3
    x = static.data(name='x', shape=[dim, dim], dtype='float32')
    
    # 使用修改后的函数来提取下三角矩阵的元素
    tril_elements = matrix_to_tril_elements(x, diagonal=-1)
    
    # 创建执行器
    exe = static.Executor(paddle.CPUPlace())
    exe.run(startup_program)
    
    # 执行图
    input_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = exe.run(main_program,
                     feed={'x': input_matrix},
                     fetch_list=[tril_elements])

    print("Elements of the lower triangular matrix:\n", result[0])
