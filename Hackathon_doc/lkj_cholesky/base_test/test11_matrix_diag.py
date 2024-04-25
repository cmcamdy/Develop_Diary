import paddle

def universal_diag_embed(diag_elems):
    # 获取diag_elems的长度，即对角线元素的数量
    num_elems = diag_elems.shape[0]
    print(num_elems)
    # 创建一个单位矩阵，然后通过广播将diag_elems扩展到对角线上
    eye_matrix = paddle.eye(num_elems, dtype=diag_elems.dtype)
    diag_matrix = eye_matrix * diag_elems.reshape([num_elems, 1])
    
    return diag_matrix

# 测试代码
paddle.disable_static()  # 测试动态图模式

diag_elems = paddle.to_tensor([1.0, 2.0, 3.0])
diag_matrix = universal_diag_embed(diag_elems)
print("Dynamic Mode:\n", diag_matrix.numpy())

paddle.enable_static()  # 测试静态图模式

# 静态图需要稍微修改测试代码
main_program = paddle.static.Program()
startup_program = paddle.static.Program()
with paddle.static.program_guard(main_program, startup_program):
    diag_elems_static = paddle.static.data(name='diag_elems', shape=[3], dtype='float32')
    diag_matrix_static = universal_diag_embed(diag_elems_static)
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_program)
    result_static = exe.run(main_program,
                            feed={'diag_elems': [1.0, 2.0, 3.0]},
                            fetch_list=[diag_matrix_static])

    print("Static Mode:\n", result_static[0])

