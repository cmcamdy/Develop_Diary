




## 代码生成
- 看到这个log
```shell
parse op yamls:
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/ops.yaml
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/legacy_ops.yaml
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/backward.yaml
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/legacy_backward.yaml
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/fused_ops.yaml
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/static_ops.yaml
validate op yaml:
- /home/cmcandy/code/PD/Paddle/paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml
- /home/cmcandy/code/PD/Paddle/paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml
```
- 出自：
paddle/fluid/operators/generator/CMakeLists.txt


### parse_op.py 
- 这里的137行开始有parse op yaml的字眼，下面应该就是解析
    - 以/home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/ops.yaml为例
    - 在这个文件里面是调用了parse_op.py来执行调用，输出到${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/ops.parsed.yaml中
```shell
COMMAND ${PYTHON_EXECUTABLE} parse_op.py --op_yaml_path ${op_yaml_file}
        --output_path ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/ops.parsed.yaml
```


- 单独测试
```shell
cd /home/cmcandy/code/PD/Paddle/paddle/fluid/operators/generator
python parse_op.py \
        --op_yaml_path /home/cmcandy/code/PD/Paddle/paddle/fluid/pir/dialect/operator/ir/ops.yaml \
        --output_path /home/cmcandy/code/PD/Develop_Diary/code_reading/ops.parsed.yaml

```



### cross_validate.py 

- 同理，validate op yaml是调用了cross_validate.py 
```shell
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} cross_validate.py --forward_yaml_paths
    ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/ops.parsed.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/legacy_ops.parsed.yaml
    --backward_yaml_paths
    ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/backward_ops.parsed.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/legacy_backward_ops.parsed.yaml
  RESULT_VARIABLE _result)
```


### generate_op.py

- 接下来生成函数：
```shell
execute_process(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/paddle/fluid/operators/generator
  COMMAND
    ${PYTHON_EXECUTABLE} generate_op.py --ops_yaml_path
    ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/ops.parsed.yaml --backward_yaml_path
    ${CMAKE_CURRENT_BINARY_DIR}/parsed_ops/backward_ops.parsed.yaml
    --op_version_yaml_path
    ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_version.yaml
    --op_compat_yaml_path ${CMAKE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml
    --output_op_path "${generated_op_path_1}.tmp" "${generated_op_path_2}.tmp"
    "${generated_op_path_3}.tmp" "${generated_op_path_4}.tmp"
    --output_arg_map_path "${generated_argument_mapping_path}.tmp"
  RESULT_VARIABLE _result)
if(${_result})
  message(FATAL_ERROR "operator codegen failed, exiting.")
endif()

```