#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"

namespace paddle {
namespace translator {
    
OpNameNormalizer::OpNameNormalizer() {
    op_name_mappings = {
    };
    op_arg_name_mappings = {
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
        { 
            "fake_quantize_dequantize_moving_average_abs_max", 
            {
                { "x", "X" },
                { "in_scale", "InScale" },
                { "in_accum", "InAccum" },
                { "in_state", "InState" },
                { "moving_rate", "moving_rate" },
                { "bit_length", "bit_length" },
                { "round_type", "round_type" },
                { "is_test", "is_test" },
                { "out", "Out" },
                { "out_scale", "OutScale" },
                { "out_state", "OutState" },
                { "out_accum", "OutAccum" },
            }, 
        },
    };
    op_mutable_attributes = {
    };
    op_mutable_attribute_infos = {
    };
}

} // namespace translator
}// namespace paddle
