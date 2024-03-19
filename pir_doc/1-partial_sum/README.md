
## 修复过程参考
https://github.com/PaddlePaddle/Paddle/issues/59382


## 老版PIR定义：parital_sum_op.cc

```
# 输入
AddInput("X", "Input tensors of partial_sum operator.").AsDuplicable();


# 输出
AddOutput("Out", "Output tensor of partial_sum operator.");

# 参数
AddAttr<int>("start_index", "The start index of tensor wanted to be added.")
        .SetDefault(0);
AddAttr<int>("length", "The length of tensor wanted to be added.")
        .SetDefault(-1);


# 注册
PD_REGISTER_STRUCT_KERNEL(partial_sum,
                          CPU,
                          ALL_LAYOUT,
                          ops::PartialSumKernel,
                          float,
                          double,
                          int,
                          int64_t) {}
```



## ops.yaml
- op定义
- paddle/fluid/pir/dialect/operator/ir/ops.yaml
- InferMeta 函数负责根据输入变量推断返回 Tensor 的维度与类型，这里是对算子使用的 InferMeta 函数进行配置

- kernel 算子的计算 Kernel 配置
```
- op : partial_sum
  args : (Tensor[] x, int start_index = 0, int length = -1)
  output : Tensor(out)
  infer_meta :
    func : PartialSumInferMeta
  kernel :
    func : sum
    data_type : x
  backward : sum_grad

```

## :?:op_compat.yaml
- ProgramTranslator需要确定应该将旧IR的哪个参数对应到新IR的哪个参数.这种映射定义在 paddle/phi/api/yaml/op_compat.yaml中.
一般地我们只需要将旧IR下对应驼峰命名转为新IR下的下划线命名即可.

- 发现这个文件下以及有定义了
```
- op : partial_sum
  backward : partial_sum_grad
  extra :
    attrs : [bool use_mkldnn = false]
```

## InferMeta

InferMeta 函数是根据输入参数，推断算子输出 Tensor 基本信息的函数，推断的信息包括输出 Tensor 的 shape、data type，同时它也承担了检查输入数据维度、类型等是否合法的功能。


修复Op单测时,并不需要我们真正去实现InferMeta,我们只需要根据需要修复Op的InferShape函数稍加修改即可,但是dtype信息需要我们单独设置一下，因为InferShape,不包含dtype信息.一般地，outputs的dtype信息要inputs的dtype一致即可.这里以dpsgd为例，介绍注册InferMeta的流程.



### 参数对应关系：

```cpp
- op : split_with_num
  args : (Tensor x, int num, Scalar(int) axis)
  output : Tensor[]{num}
  infer_meta :
    func : SplitWithNumInferMeta
    spmd_rule : SplitWithNumInferSpmdDynamic
  kernel :
    func : split_with_num
  backward : split_with_num_grad
  interfaces : paddle::dialect::InferSymbolicShapeInterface

void SplitWithNumInferMeta(const MetaTensor& x,
                           int num,
                           const Scalar& axis,
                           std::vector<MetaTensor*> out,
                           MetaConfig config) 

```


- backward_op : partial_sum_grad
  forward : partial_sum (Tensor[] x, int start_index = 0, int length = -1) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, int start_index = 0, int length = -1)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
    spmd_rule : ReductionGradInferSpmd
  kernel :
    func : sum_grad
  composite : sum_grad(x, out_grad, axis, keepdim, reduce_all, x_grad)
  no_need_buffer : x
  backward : sum_double_grad





  load_op_meta_info_and_register_op -> LoadOpMetaInfoAndRegisterOp > RegisterOperatorWithMetaInfoMap 

  RegisterAllCustomOperator


  cp ~/code/PD/Paddle/paddle/fluid/pir/dialect/op_generator/ops_api_gen.py .
  cp ~/code/PD/Paddle/paddle/fluid/pir/dialect/op_generator/ops_api_gen.py
  cp ~/code/PD/Paddle/paddle/fluid/pir/dialect/operator/ir/ops.yaml
  cp ~/code/PD/Paddle/paddle/fluid/pir/dialect/operator/ir/ops_backward.yaml
  cp ~/code/PD/Paddle/paddle/fluid/pir/dialect/operator/utils/utils.cc
  cp ~/code/PD/Paddle/paddle/phi/api/yaml/op_compat.yaml
  cp ~/code/PD/Paddle/paddle/phi/infermeta/backward.cc
  cp ~/code/PD/Paddle/paddle/phi/infermeta/backward.h
  cp ~/code/PD/Paddle/paddle/phi/infermeta/unary.cc
  cp ~/code/PD/Paddle/paddle/phi/infermeta/unary.h
  cp ~/code/PD/Paddle/paddle/phi/infermeta/unary.h

          
        





