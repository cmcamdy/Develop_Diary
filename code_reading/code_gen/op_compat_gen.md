






## 概览
/home/cmcandy/code/PD/Paddle/paddle/fluid/ir_adaptor/translator/CMakeLists.txt
- 首先需要看这个CMakeLists中的内容，
    - **即调用op_compat_gen.py，读取op_compat.yaml，并分析，生成op_compat_info.cc**
```cpp
set(PD_PROGRAM_TRANSLATOR_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(PD_PROGRAM_TRANSLATOR_BINARY_DIR
    "${PADDLE_BINARY_DIR}/paddle/fluid/ir_adaptor/translator/")

set(op_gen_file ${PD_PROGRAM_TRANSLATOR_SOURCE_DIR}/op_compat_gen.py)
set(op_compat_yaml_file ${PADDLE_SOURCE_DIR}/paddle/phi/api/yaml/op_compat.yaml)
set(op_compat_source_file ${PD_PROGRAM_TRANSLATOR_SOURCE_DIR}/op_compat_info.cc)
set(op_compat_templat_file
    ${PD_PROGRAM_TRANSLATOR_SOURCE_DIR}/op_compat_info.cc.j2)

add_custom_command(
  OUTPUT ${op_compat_source_file}
  COMMAND ${PYTHON_EXECUTABLE} ${op_gen_file} --op_compat_yaml_file
          ${op_compat_yaml_file} --output_source_file ${op_compat_source_file}
  DEPENDS ${op_gen_file} ${op_compat_yaml_file} ${op_compat_templat_file}
  VERBATIM)

file(GLOB PD_PROGRAM_TRANSLATOR_SRCS "*.cc")

cc_library(
  program_translator
  SRCS ${PD_PROGRAM_TRANSLATOR_SRCS} ${op_compat_source_file}
  DEPS proto_desc op_dialect op_dialect_vjp pir framework_proto)
x
```




### op_compat_info.cc
/home/cmcandy/code/PD/Paddle/paddle/fluid/ir_adaptor/translator/op_compat_info.cc
- 这里面存的是
```
op_name_mappings
op_arg_name_mappings
op_mutable_attributes
op_mutable_attribute_infos
```
- 乍一看是映射，全局搜索一下op_name_mappings就可以发现这个文件：
    - /home/cmcandy/code/PD/Paddle/paddle/fluid/ir_adaptor/translator/op_compat_info.h
- 这里面OpNameNormalizer有这么个东西，也就是定义，op_compat_info.cc这个就是实现了一下它的无参构造函数，这个东西是通过上文的CMakeLists.txt实现，也就是op_compat_gen.py分析op_compat.yaml并进行CodeGen
```cpp

class OpNameNormalizer {
 private:
  OpNameNormalizer();  // Disallow instantiation outside of the class.
  std::unordered_map<std::string, std::string> op_name_mappings;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      op_arg_name_mappings;

  std::unordered_map<std::string,
                     std::unordered_map<std::string, MutableAttributeInfo>>
      op_mutable_attribute_infos;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      op_mutable_attributes;
.......
};

```

- 这个东西的作用是什么？
    - 记录一些特殊转换，
        - 如：op_name_mappings记录的大概率是OP的名称转换
        - 如：op_arg_name_mappings记录的大概率是算子某些参数的名称转换


### 如何配置op_compat.yaml
- /home/cmcandy/code/PD/Paddle/paddle/phi/api/yaml/op_compat.yaml

```
- op : partial_sum
  backward : partial_sum_grad
  inputs :
    x : X
  outputs :
    out : Out
  extra :
    attrs : [bool use_mkldnn = false]
```

- 这个玩意会通过op_compat_gen.py生成:
```
....
op_arg_name_mappings = {
    ....
    { 
        "partial_sum", 
        {
            { "x", "X" },
            { "out", "Out" },
        }, 
    },
    { 
        "partial_sum_grad", 
        {
            { "x", "X" },
            { "out", "Out" },
        }, 
    },
    ....
}
....
```


### op_compat_gen.py的主体逻辑
- /home/cmcandy/code/PD/Paddle/paddle/fluid/ir_adaptor/translator/op_compat_gen.py

- 主要是这个函数:
```
def OpNameNormalizerInitialization(
    op_compat_yaml_file: str = "", output_source_file: str = ""
) -> None:
....
```