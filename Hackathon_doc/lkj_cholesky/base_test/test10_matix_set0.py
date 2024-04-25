import paddle
import paddle.static as static

# 启用静态图模式
paddle.enable_static()

# 定义数据
data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

# 创建静态图程序
main_program = static.Program()
startup_program = static.Program()

# 使用静态图的 program
with static.program_guard(main_program, startup_program):
    # 创建一个变量，代表一个将要输入的矩阵
    x = static.data(name='x', shape=[3, 2], dtype='float32')
    
    # 创建一个新的张量，其第0行为0，其余行与x相同
    zero_row = paddle.zeros(shape=[1, 2], dtype='float32')
    other_rows = x[1:, :]
    new_x = paddle.concat([zero_row, other_rows], axis=0)

    # 创建执行器，用于运行静态图程序
    place = paddle.CPUPlace()
    exe = static.Executor(place)

    # 运行初始化程序
    exe.run(startup_program)

    # 运行主程序
    output = exe.run(main_program,
                     feed={'x': data},
                     fetch_list=[new_x])

    print("Original Matrix:")
    print(data)
    print("Modified Matrix:")
    print(output[0])
