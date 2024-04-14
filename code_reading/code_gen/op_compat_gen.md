






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

```

op_gen_file=/home/cmcandy/code/PD/Develop_Diary/code_reading/code_gen/test_op_compat/op_compat_gen.py
op_compat_yaml_file=/home/cmcandy/code/PD/Develop_Diary/code_reading/code_gen/test_op_compat/op_compat.yaml
op_compat_source_file=/home/cmcandy/code/PD/Develop_Diary/code_reading/code_gen/test_op_compat/op_compat_info.cc
op_compat_templat_file=/home/cmcandy/code/PD/Develop_Diary/code_reading/code_gen/test_op_compat/op_compat_info.cc.j2
python ${op_gen_file}  --op_compat_yaml_file ${op_compat_yaml_file} --output_source_file ${op_compat_source_file}
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

- 主要逻辑
```
# yaml都进来，dict数组
with open(op_compat_yaml_file, "r") as f:
        op_compat_infos = yaml.safe_load(f)
```
- 后有for循环，逐个遍历op_compat_infos，其中有三个逻辑：
- **insert_new_mappings、insert_new_arg_mappings、insert_new_mutable_attributes**
- 下面是GPT的功能解释：

1. `insert_new_mappings(op_name_str: str) -> str`

    - 这个函数用于插入新的映射关系。输入参数 `op_name_str` 是一个字符串，表示操作的名字。函数内部首先调用 `to_phi_and_fluid_op_name(op_name_str)`，得到两个名字：`normalized_name` 和 `legacy_name`。如果这两个名字相同，则直接返回它们。否则，将键值对 `{legacy_name: normalized_name}` 添加到字典 `op_name_mappings` 中，并返回这两个名字。这个函数主要用于处理操作名的映射关系。

2. `insert_new_arg_mappings(op_name: str, arg_mapping: Dict[str, str])`

    - 这个函数用于插入新的参数映射关系。输入参数 `op_name` 是一个字符串，表示操作的名字；`arg_mapping` 是一个字典，表示参数的映射关系。函数首先检查 `op_name` 是否存在于字典 `op_arg_name_mappings` 中，如果不存在，则为 `op_name` 创建一个空字典。然后，使用 `update()` 方法将 `arg_mapping` 更新到 `op_arg_name_mappings[op_name]` 中。这个函数主要用于处理操作参数的映射关系。

3. `insert_new_mutable_attributes(op_name: str, mutable_attribute_infos: Dict[str, Dict[str, str]])`

    - 这个函数用于插入新的可变属性信息。输入参数 `op_name` 是一个字符串，表示操作的名字；`mutable_attribute_infos` 是一个嵌套字典，表示可变属性的信息。函数首先检查 `op_name` 是否存在于字典 `op_mutable_attributes` 和 `op_mutable_attribute_infos` 中，如果不存在，则为 `op_name` 创建一个空集合和一个空字典。接着，遍历 `mutable_attribute_infos`，将属性名添加到 `op_mutable_attributes[op_name]` 中，并将属性信息添加到 `op_mutable_attribute_infos[op_name]` 中。这个函数主要用于处理操作的可变属性信息。

**那么主体逻辑就是**：

这段代码逻辑主要用于处理 `op_compat_item` 中的映射关系和属性。`op_compat_item` 是一个字典，包含了操作的兼容性信息。下面逐步解释这段逻辑：
1. 首先，调用 `insert_new_mappings(op_compat_item["op"])` 函数，获取操作的映射关系。`legacy_name` 是旧的操作名。

2. 初始化一个空列表 `legacy_backward_op_names`，用于存储反向操作的旧名字。如果 `op_compat_item` 中包含 `"backward"` 键，那么将反向操作名的映射关系添加到 `legacy_backward_op_names` 列表中。

3. 如果 `op_compat_item` 中包含 `"inputs"` 键，那么调用 `insert_new_arg_mappings()` 函数，将输入参数的映射关系添加到 `legacy_name` 和 `legacy_backward_op_names` 中。

4. 如果 `op_compat_item` 中包含 `"attrs"` 键，那么调用 `insert_new_arg_mappings()` 函数，将属性参数的映射关系添加到 `legacy_name` 和 `legacy_backward_op_names` 中。

5. 如果 `op_compat_item` 中包含 `"outputs"` 键，那么调用 `insert_new_arg_mappings()` 函数，将输出参数的映射关系添加到 `legacy_name` 和 `legacy_backward_op_names` 中。

6. 如果 `op_compat_item` 中包含 `"int_array"` 键，那么调用 `insert_new_mutable_attributes()` 函数，将整数数组类型的可变属性信息添加到 `legacy_name` 和 `legacy_backward_op_names` 中。

7. 如果 `op_compat_item` 中包含 `"scalar"` 键，那么调用 `insert_new_mutable_attributes()` 函数，将标量类型的可变属性信息添加到 `legacy_name` 和 `legacy_backward_op_names` 中。


### 疑问
- 在有些配置中，有extra项，但是这里没有解析到，可能是在别的地方有用？TODO
```
  extra :
    attrs : [int round_type = 1]
```