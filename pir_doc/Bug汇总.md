

## Bug汇总


### RuntimeError: (PreconditionNotMet) op [pd_op.xxx] kernel output args defs should equal op outputs
- 此类问题是Legacy op的kernel和phi kernel的推导机制不一致造成的。如果kernel是通过PD_REGISTER_STRUCT_KERNEL注册的，需要把他加在LegacyOpList中，单独处理。

### PreconditionNotMetError: Tensor holds no memory. Call Tensor::mutable_data firstly.

- 此类问题一般是feed/fetch/memcpy_d2h...等算子造成,执行之前需要初始化一下所在空间如:
- `out_scale->mutable_data<T>(context.GetPlace());`

### FatalError: `Process abort signal` is detected by the operating system.
- 第一次遇到是InferMeta写错了,导致一个参数的dim没分配,然后非法访问

### NotFoundError: (xxx) is not found in AttributeMap and RuntimeAttributeMap of (xxx) operator.
- 这个可能是老版本的定义漏了,不知道为啥能过编译
    - 这个是OPMaker的attr定义漏了。。。。

### ValueError: (InvalidArgument) Input xxx not found when parsing op xxxx
- 这个一般是op_compat.yaml没配置好,要么就是有特殊规则,需要在op_compat_gen.py中单独指定