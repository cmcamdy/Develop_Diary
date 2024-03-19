## PD_REGISTER_STRUCT_KERNEL

- 这个宏的定义是：
```
#define PD_REGISTER_STRUCT_KERNEL(                            \
    kernel_name, backend, layout, meta_kernel_structure, ...) \
  _PD_REGISTER_KERNEL(::phi::RegType::INNER,                  \
                      kernel_name,                            \
                      backend,                                \
                      ::phi::backend##Context,                \
                      layout,                                 \
                      meta_kernel_structure,                  \
                      STRUCTURE_KERNEL_INSTANTIATION,         \
                      STRUCTURE_ARG_PARSE_FUNCTOR,            \
                      PHI_STRUCTURE_KERNEL,                   \
                      PHI_STRUCTURE_VARIADIC_KERNEL,          \
                      __VA_ARGS__)
```

## REGISTER_OPERATOR

```cpp
/*
  The variadic arguments should be class types derived from one of the
  following classes:
    OpProtoAndCheckerMaker
    GradOpDescMakerBase
    VarTypeInference
    InferShapeBase
*/
#define REGISTER_OPERATOR(op_type, op_class, ...)                        \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                        \
      __reg_op__##op_type,                                               \
      "REGISTER_OPERATOR must be called in global namespace");           \
  static ::paddle::framework::OperatorRegistrar<op_class, ##__VA_ARGS__> \
      __op_registrar_##op_type##__(#op_type);                            \
  int TouchOpRegistrar_##op_type() {                                     \
    __op_registrar_##op_type##__.Touch();                                \
    return 0;                                                            \
  }

```


## REGISTER_OP_WITHOUT_GRADIENT

```cpp
#define REGISTER_OP_WITHOUT_GRADIENT(op_type, op_class, ...) \
  REGISTER_OPERATOR(op_type, op_class, __VA_ARGS__, \
        paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,   \
        paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

```