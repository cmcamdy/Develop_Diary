import numpy as np
import paddle
import paddle.static as static

paddle.enable_static()  # 启用静态图模式

# 创建静态图程序
main_program = static.Program()
startup_program = static.Program()

with static.program_guard(main_program, startup_program):
    matrix = static.data(name='matrix', shape=[3, 3], dtype='float32')
    indices = static.data(name='indices', shape=[2, 2], dtype='int32')
    updates = static.data(name='updates', shape=[2], dtype='float32')
    output = paddle.scatter_nd_add(matrix, indices, updates)

exe = static.Executor(paddle.CPUPlace())
exe.run(startup_program)

# 模拟输入数据
matrix_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
indices_data = np.array([[0, 0], [1, 1]], dtype=np.int32)
updates_data = np.array([10, 20], dtype=np.float32)

# 执行计算
result = exe.run(main_program,
                 feed={'matrix': matrix_data, 'indices': indices_data, 'updates': updates_data},
                 fetch_list=[output])

print(result[0])
