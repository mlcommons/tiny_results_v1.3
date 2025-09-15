#pragma once

#define WEIGHT_QUANTIZATION_GRANULARITY 0  // 0 means per_tensor, 1 means per_channel
#define ACTIVATION_QUANTIZATION_TYPE 1  // 0 means int8, 1 means uint8
#define NODE_NUM 32
#define INPUT_NUM 32
#define INPUT2NODES_LEN 1
#define NODE2INPUTS_LEN 1
#define ACTIVATIONS_NUM 33
#define CONV2D_OP
#define DEPTHWISE_CONV2D_OP
#define MATMUL_OP
#define PERMUTE_OP
#define VIEW_OP
#define SOFTMAX_OP
#define AVERAGE_POOL_OP
extern const unsigned char IMAGE_DATA[];
extern const int IMAGE_DATA_DIMS[];
extern const int NODE_TYPE[];
extern const char WEIGHT_INT8[];
extern const int WEIGHT_INT32[];
extern const int CONFIG[];
extern const int QUANTIZATION_FACTOR[];

extern const int NODE2OUTPUTS[NODE_NUM][1];
extern const int NODE2INPUTS[NODE_NUM][NODE2INPUTS_LEN];
extern const int INPUT2NODES[INPUT_NUM][INPUT2NODES_LEN];
extern const int AVERAGE_POOLING_DIVISOR[]; /* average pooling divisor */
extern const int CONV_WEIGHT_POS[];
extern const int CONV_BIAS_POS[];
extern const int CONV_WEIGHT_SHAPE_POS[];
extern const int CONV_BIAS_SHAPE_POS[];
extern const int CONV_QUANTIZATION_FACTOR_POS[];
extern const int CONV_STRIDE_POS[];
extern const int CONV_PAD_POS[];
extern const int CONV_ACTIVATION_FUNC_POS[];

extern const int MATMUL_WEIGHT_POS[];
extern const int MATMUL_BIAS_POS[];
extern const int MATMUL_WEIGHT_SHAPE_POS[];
extern const int MATMUL_BIAS_SHAPE_POS[];
extern const int MATMUL_QUANTIZATION_FACTOR_POS[];
extern const int MATMUL_TRANSA_POS[];
extern const int MATMUL_TRANSB_POS[];
extern const int MATMUL_ACTIVATION_FUNC_POS[];

extern const int TRANSPOSE_PERM_POS[];

extern const int RESHAPE_SHAPE_POS[];

extern const int SOFTMAX_AXIS_POS[];

extern const int AVERAGE_POOL_KERNEL_SHAPE_POS[];
extern const int AVERAGE_POOL_STRIDES_POS[];

