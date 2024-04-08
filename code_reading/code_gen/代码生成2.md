有关于pir下面的生成代码在这个CMakeLists
/home/cmcandy/code/PD/Paddle/paddle/fluid/pir/dialect/CMakeLists.txt

```
set(PD_DIALECT_SOURCE_DIR
    "${PADDLE_BINARY_DIR}/paddle/fluid/pir/dialect/operator/ir")
```

## yaml 文件
- 参数定义：
```yaml
op_gen_parsed_yaml_file     paddle/fluid/operators/generator/parse_op.py

op_gen_file                 paddle/fluid/pir/dialect/op_generator/op_gen.py

op_compat_yaml_file         paddle/phi/api/yaml/op_compat.yaml

op_fwd_yaml                 paddle/fluid/operators/generator/parsed_ops/ops.parsed.yaml

op_bwd_yaml                 paddle/fluid/operators/generator/parsed_ops/backward_ops.parsed.yaml

fused_op_fwd_yaml           paddle/fluid/operators/generator/parsed_ops/fused_ops.parsed.yaml

fused_op_bwd_yaml           paddle/fluid/operators/generator/parsed_ops/fused_backward.parsed.yaml

pir_op_fwd_src_yaml         paddle/fluid/pir/dialect/operator/ir/ops.yaml

pir_op_bwd_src_yaml         paddle/fluid/pir/dialect/operator/ir/ops_backward.yaml

pir_update_op_fwd_src_yaml  paddle/fluid/pir/dialect/operator/ir/update_ops.yaml

# /home/cmcandy/code/PD/Paddle/build/paddle/fluid/pir/dialect/operator/ir/generated
# 这里记录着生成yaml的目录，里面有生成好的pir的yaml：ops.parsed.yaml、ops_backward.parsed.yaml
parsed_op_dir               paddle/fluid/pir/dialect/operator/ir/generated

pir_op_fwd_yaml ${parsed_op_dir}/ops.parsed.yaml

pir_op_bwd_yaml ${parsed_op_dir}/ops_backward.parsed.yaml

pir_update_op_fwd_yaml ${parsed_op_dir}/update_ops.parsed.yaml
```




### Auto CodeGen

```shell
# Auto code gen
execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${parsed_op_dir}

    COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
            ${pir_op_fwd_src_yaml} --output_path ${pir_op_fwd_yaml}

    # 等价于：
    # python paddle/fluid/operators/generator/parse_op.py --op_yaml_path paddle/fluid/pir/dialect/operator/ir/ops.yaml \
    #           --output_path paddle/fluid/pir/dialect/operator/ir/generated/ops.parsed.yaml

    COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
            ${pir_update_op_fwd_src_yaml} --output_path ${pir_update_op_fwd_yaml}
    
    # 等价于：
    # python paddle/fluid/operators/generator/parse_op.py --op_yaml_path paddle/fluid/pir/dialect/operator/ir/update_ops.yaml \
    #           --output_path paddle/fluid/pir/dialect/operator/ir/generated/update_ops.parsed.yaml

    COMMAND ${PYTHON_EXECUTABLE} ${op_gen_parsed_yaml_file} --op_yaml_path
            ${pir_op_bwd_src_yaml} --output_path ${pir_op_bwd_yaml} --backward)
    
    # 等价于：
    # python paddle/fluid/operators/generator/parse_op.py --op_yaml_path paddle/fluid/pir/dialect/operator/ir/ops_backward.yaml \
    #           --output_path paddle/fluid/pir/dialect/operator/ir/generated/ops_backward.parsed.yaml
    

execute_process(
  COMMAND
    ${PYTHON_EXECUTABLE} ${op_gen_file} --op_yaml_files ${op_yaml_files}
    --op_compat_yaml_file ${op_compat_yaml_file} --namespaces ${op_namespace}
    --dialect_name ${dialect_name} --op_def_h_file ${op_header_file_tmp}
    --op_info_file ${op_info_file_tmp} --op_def_cc_file ${op_src_files_tmp}
    --op_vjp_cc_file ${op_vjp_src_file_tmp} --with_distributed
    ${WITH_DISTRIBUTE})

    # 等价于
    python paddle/fluid/pir/dialect/op_generator/op_gen.py  \
                --op_yaml_files ${op_yaml_files}    \
                --op_compat_yaml_file paddle/phi/api/yaml/op_compat.yaml \
                --namespaces paddle,dialect  \
                --dialect_name pd_op        \
                --op_def_h_file     paddle/fluid/pir/dialect/operator/ir/pd_op.h.tmp    \
                --op_info_file      paddle/fluid/pir/dialect/operator/ir/pd_op_info.cc.tmp  \
                --op_def_cc_file    ${op_src_files_tmp}    \
                --op_vjp_cc_file    paddle/fluid/pir/dialect/operator/ir/pd_op_vjp.cc.tmp 
                --with_distributed ${WITH_DISTRIBUTE}
```

- 其中
```cpp
op_yaml_files：
        ${op_fwd_yaml},                 ops.parsed.yaml
        ${op_bwd_yaml},                 backward_ops.parsed.yaml
        ${fused_op_fwd_yaml},           fused_ops.parsed.yaml
        ${fused_op_bwd_yaml},           fused_backward.parsed.yaml
        ${pir_op_fwd_yaml},             ops.parsed.yaml
        ${pir_op_bwd_yaml},             ops_backward.parsed.yaml
        ${pir_update_op_fwd_yaml}       update_ops.parsed.yaml

op_src_files_tmp：
    ${op_source_file_tmp},              pd_op.cc.tmp
    ${bwd_op_source_file_tmp},          pd_op_bwd.cc.tmp
    ${fused_op_source_file_tmp},        pd_op_fused.cc.tmp
    ${bwd_fused_op_source_file_tmp},    pd_op_fused_bwd.cc.tmp
    ${pir_op_source_file_tmp},          pd_pir_op.cc.tmp
    ${pir_bwd_op_source_file_tmp},      pd_pir_op_bwd.cc.tmp
    ${pir_update_op_source_file_tmp}    pd_pir_op_update.cc.tmp

```

