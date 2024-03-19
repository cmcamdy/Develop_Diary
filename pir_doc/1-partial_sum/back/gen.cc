std::vector<pir::Type> PartialSumGradOp::InferMeta(const std::vector<pir::Value>& input_values, const pir::AttributeMap& attributes) {

  IR_ENFORCE(input_values.size() == 2,
      "Num of inputs is expected to be 2 but got %d.", input_values.size());

  pir::Value x_ = input_values[0]; (void)x_;
  VLOG(4) << "Builder construction outputs";
  bool is_from_tensor = false; (void) is_from_tensor;
  pir::VectorType x = x_.type().dyn_cast<pir::VectorType>(); (void)x;


  std::vector<paddle::dialect::IrTensor> vec_ir_tensor_x;
  for (size_t i=0; i < static_cast<size_t>(x.size()); i++) {
    if(x[i].isa<paddle::dialect::DenseTensorType>()) {
        auto x_type = x[i].dyn_cast<paddle::dialect::DenseTensorType>();
        vec_ir_tensor_x.push_back(paddle::dialect::IrTensor(paddle::dialect::TransToPhiDataType(x_type.dtype()),
                                                                    x_type.dims(),
                                                                    x_type.data_layout(),
                                                                    x_type.lod(),
                                                                    x_type.offset()));
    } else {
        PADDLE_THROW(phi::errors::Unimplemented("Only support DenseTensorType or AllocatedDenseTensorType"));
    }
  }
  std::vector<paddle::dialect::IrMetaTensor> vec_meta_x;
  for (size_t i=0; i < vec_ir_tensor_x.size(); i++) {
    vec_meta_x.push_back(paddle::dialect::IrMetaTensor(&vec_ir_tensor_x[i]));
  }

  std::vector<const phi::MetaTensor*> meta_x;
  for (size_t i=0; i < static_cast<size_t>(vec_meta_x.size()); i++) {
    meta_x.push_back(&vec_meta_x[i]);
  }
   paddle::dialect::IrTensor dense_x_grad;
  paddle::dialect::IrMetaTensor meta_x_grad(&dense_x_grad);

  phi::GeneralUnaryGradInferMeta(meta_x, &meta_x_grad);

  std::vector<pir::Type> argument_outputs;
  pir::Type x_grad_dense_tensor_type = paddle::dialect::DenseTensorType::get(pir::IrContext::Instance(), paddle::dialect::TransToIrDataType(dense_x_grad.dtype()), dense_x_grad.dims(), dense_x_grad.layout(), dense_x_grad.lod(), dense_x_grad.offset());
  argument_outputs.push_back(x_grad_dense_tensor_type);

  return argument_outputs;
}
