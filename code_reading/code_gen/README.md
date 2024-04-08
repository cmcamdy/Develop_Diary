## 文件说明

- Develop_Diary/code_reading/code_gen/test_yaml下存的是原始构造的yaml
- Develop_Diary/code_reading/code_gen/test_yaml_parsed下存的是经过pased之后的yaml
- Develop_Diary/code_reading/code_gen/test_codegen下存的是pir生成的代码

## 问题的复现

- 首先需要一个paddle仓库，相关的cmake逻辑如果没有变动，应当在这里：paddle/fluid/pir/dialect/CMakeLists.txt
- 有关于这个CMakeLists的阅读记录在：Develop_Diary/code_reading/code_gen/代码生成2.md
    - 总而言之，这里的逻辑就是用几个python脚本分析yaml，然后进行codegen（字符串拼接）

### 模拟生成
- 生成的脚本记录在：Develop_Diary/code_reading/code_gen/测试脚本.md
    - 读者如需要复现，可能得费神修改下地址（参考文件说明中的三个地址）

### 问题复现

- 如果配置重复的名字（如当前的test_yaml/ops_backward.yaml中的两个straight_through_estimator_grad），则在运行op_gen.py的时候会卡在这里：
```shell
straight_through_estimator_grad ['x', 'in_accum', 'in_state'] 4
Traceback (most recent call last):
  File "/home/cmcandy/code/PD/Paddle/paddle/fluid/pir/dialect/op_generator/op_gen.py", line 2349, in <module>
    OpGenerator(
  File "/home/cmcandy/code/PD/Paddle/paddle/fluid/pir/dialect/op_generator/op_gen.py", line 2164, in OpGenerator
    ) = AutoCodeGen(
        ^^^^^^^^^^^^
  File "/home/cmcandy/code/PD/Paddle/paddle/fluid/pir/dialect/op_generator/op_gen.py", line 1310, in AutoCodeGen
    input_grad_semantics = get_input_grad_semantic(
                           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/cmcandy/code/PD/Paddle/paddle/fluid/pir/dialect/op_generator/op_gen.py", line 1105, in get_input_grad_semantic
    len(bwd_fwd_input_list) == num_inputs
AssertionError: Configuration of forward op and backward op is not match.
```
- 具体是下面的这个assert:``len(bwd_fwd_input_list) == num_inputs``

```python
def get_input_grad_semantic(op_info, op_info_items):
    input_grad_semantics = []
    num_inputs = len(op_info.input_name_list)

    # get backward op
    bwd_op_name = op_info.backward_name
    if (bwd_op_name is None) or (bwd_op_name not in op_info_items.keys()):
        input_grad_semantics = ["false" for i in range(num_inputs)]
    else:
        bwd_op_info = op_info_items[bwd_op_name]

        # cut "_grad" of each output of bwd_op, and then compare each modified output with corresponding input
        # thus determine whether each input has grad semantic
        bwd_output_list = bwd_op_info.output_name_list
        bwd_output_list_new = []
        for bwd_output in bwd_output_list:
            bwd_output_list_new.append(bwd_output[:-5])  # cut _grad

        bwd_fwd_input_list = bwd_op_info.forward_input_name_list
        if bwd_fwd_input_list is not None:
            print(bwd_op_name, bwd_fwd_input_list, num_inputs)
            assert (
                len(bwd_fwd_input_list) == num_inputs
            ), "Configuration of forward op and backward op is not match."
            for i in range(num_inputs):
                if bwd_fwd_input_list[i] in bwd_output_list_new:
                    input_grad_semantics.append("true")
                else:
                    input_grad_semantics.append("false")
        else:
            input_grad_semantics = ["false" for i in range(num_inputs)]

    return input_grad_semantics
```

- 我打印了下相关参数：
    - `print(bwd_op_name, bwd_fwd_input_list, op_info.input_name_list, num_inputs)`
    - 结果为：`straight_through_estimator_grad ['x', 'in_accum', 'in_state'] ['x', 'in_scale', 'in_accum', 'in_state'] 4`
    - 出现这个问题的原因可从`Develop_Diary/code_reading/code_gen/test_yaml_parsed/ops_backward.pased.yaml`这个文件中的两个straight_through_estimator_grad分析的出：
        - moving_average_abs_max_scale和fake_quantize_dequantize_moving_average_abs_max的input分别对应上述这两沱参数名，目前看来应该是记录参数的时候先记录fake_quantize_dequantize_moving_average_abs_max的grad函数，后记录的moving_average_abs_max_scale的grad由于key相同，覆盖了前者，这才导致bwd_fwd_input_list是后者的数量
    
    
### 存在的疑问
- backward中的name可以自定义吗，他和kernel-func之间是什么关系呢？