

## fake_quantize_moving_average_abs_max的IR图
- fake_quantize_moving_average_abs_max这个算子的IR图
    - 看起来是用feed将几个输入tensor包起来，fetch将几个输出取出来
    - TODO：feed、fetch的作用

```cpp
======================== The network executed by pir interpreter ========================
{
    (%0) = "(phi_kernel)" () {col:(Int32)3,is_persistable:[false],kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"",name:"InState",op_name:"pd_op.feed",stop_gradient:[false]} : () -> cpu_tensor<1xf32>
    (%1) = "(phi_kernel)" () {col:(Int32)2,is_persistable:[false],kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"",name:"InAccum",op_name:"pd_op.feed",stop_gradient:[false]} : () -> cpu_tensor<1xf32>
    (%2) = "(phi_kernel)" () {col:(Int32)1,is_persistable:[false],kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"",name:"InScale",op_name:"pd_op.feed",stop_gradient:[false]} : () -> cpu_tensor<1xf32>
    (%3) = "(phi_kernel)" () {col:(Int32)0,is_persistable:[false],kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"",name:"X",op_name:"pd_op.feed",stop_gradient:[false]} : () -> cpu_tensor<8x16x7x7xf32>
    
    (%4, %5, %6, %7) = "fake_quantize_moving_average_abs_max(legacy_kernel)" (%3, %2, %1, %0) {bit_length:(Int32)5,is_persistable:[false,false,false,false],is_test:false,kernel_key:<backend:CPU|layout:NCHW|dtype:float32>,kernel_name:"fake_quantize_moving_average_abs_max",moving_rate:(Float)0.9,op_name:"pd_op.fake_quantize_moving_average_abs_max",round_type:(Int32)0,stop_gradient:[false,false,false,false]} : (cpu_tensor<8x16x7x7xf32>, cpu_tensor<1xf32>, cpu_tensor<1xf32>, cpu_tensor<1xf32>) -> cpu_tensor<8x16x7x7xf32>, cpu_tensor<1xf32>, cpu_tensor<1xf32>, cpu_tensor<1xf32>
    
    (%8) = "fetch(phi_kernel)" (%4) {col:(Int32)0,kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"fetch",name:"Out",op_name:"pd_op.fetch"} : (cpu_tensor<8x16x7x7xf32>) -> cpu_tensor<8x16x7x7xf32>
    (%9) = "fetch(phi_kernel)" (%7) {col:(Int32)1,kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"fetch",name:"OutAccum",op_name:"pd_op.fetch"} : (cpu_tensor<1xf32>) -> cpu_tensor<1xf32>
    (%10) = "fetch(phi_kernel)" (%6) {col:(Int32)2,kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"fetch",name:"OutState",op_name:"pd_op.fetch"} : (cpu_tensor<1xf32>) -> cpu_tensor<1xf32>
    (%11) = "fetch(phi_kernel)" (%5) {col:(Int32)3,kernel_key:<backend:CPU|layout:Undefined(AnyLayout)|dtype:float32>,kernel_name:"fetch",name:"OutScale",op_name:"pd_op.fetch"} : (cpu_tensor<1xf32>) -> cpu_tensor<1xf32>
}

```

- 其实在看的时候可以读这个，比较简洁
- 可以比较清晰地看出：fake_quantize_moving_average_abs_max这个算子的输入为0 1 2 3，返回4 5 6 7，其中4 5 6 7是中间地址，被fetch到输出地址8 9 10 11
```cpp
======================== The instruction executed by pir interpreter ========================
{outputs} =  instruction_name[idx] ({inputs})
0: ( 7 ) ( 6 ) ( 5 ) ( 4 )  = pd_op.fake_quantize_moving_average_abs_max ( 0 )  ( 1 )  ( 2 )  ( 3 ) 
1: ( 8 )  = pd_op.fetch ( 4 ) 
2: ( 9 )  = pd_op.fetch ( 7 ) 
3: ( 10 )  = pd_op.fetch ( 6 ) 
4: ( 11 )  = pd_op.fetch ( 5 ) 
---------------------------var_id -> var_name -> variable*---------------------------
0 -> InState -> 0x6d7cdc0
1 -> InAccum -> 0x6d7bdd0
2 -> InScale -> 0x6d7c270
3 -> X -> 0x6d7e260
4 -> 0x6d7f0c01712805768044054927_inner_var_4 -> 0x6d7cae0
5 -> 0x6d7f0c01712805768044054927_inner_var_5 -> 0x6d7db20
6 -> 0x6d7f0c01712805768044054927_inner_var_6 -> 0x6d7d6c0
7 -> 0x6d7f0c01712805768044054927_inner_var_7 -> 0x6d4b0f0
8 -> Out@fetch -> 0x6d4b550
9 -> OutAccum@fetch -> 0x6d4b810
10 -> OutState@fetch -> 0x6d4baa0
11 -> OutScale@fetch -> 0x6d4bd80
```

- 这里输出了一些依赖关系
```cpp
======================= The dependency of all instruction ========================
id -> down_stream_id
0 -> 1 2 3 4 

I0411 11:22:48.046643 21284 pir_interpreter.cc:636] Analyze the execution order of Trace scheduling mode.
I0411 11:22:48.046648 21284 pir_interpreter.cc:640] op_id: 1, remain deps: 1
I0411 11:22:48.046651 21284 pir_interpreter.cc:640] op_id: 2, remain deps: 1
I0411 11:22:48.046654 21284 pir_interpreter.cc:640] op_id: 3, remain deps: 1
I0411 11:22:48.046656 21284 pir_interpreter.cc:640] op_id: 4, remain deps: 1
======================== pir interpreter trace order ========================

Leaf nodes: 0[pd_op.fake_quantize_moving_average_abs_max]->
0 downstreams: 1[pd_op.fetch]->2[pd_op.fetch]->3[pd_op.fetch]->4[pd_op.fetch]->
1 downstreams: 
2 downstreams: 
3 downstreams: 
4 downstreams: 
```