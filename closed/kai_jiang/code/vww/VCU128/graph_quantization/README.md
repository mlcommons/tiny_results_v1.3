# Model Parsing Tool

This is the model parsing tool for model parsing, model quantization, and model serialization to generate quantized model data files. The scripts for parsing model are ```generate_quantized_model_ic.sh``` and ```generate_quantized_model_vww.sh```.
In ```generate_quantized_model_ic.sh```, the ```path to samples_uint8 pickle file``` means the data file used for quantization calibration in the **ic** benchmark, which is generated in ```code/ic/LC236R5E176_BOARD_V2.0/dataset/generate_sample_data.sh```. 

In ```generate_quantized_model_vww.sh```, the ```path to calibration_official``` means the data file used for quantization calibration in the **vww** benchmark, which uses the officially provided calibration data.


## Explanation About Output Model Data

In the **ic** benchmark，the output model data are located in ```temps/ic```. 
In the **vww** benchmark，the output model data are located in ```temps/vww```. 

The model parsing tool can generate model data with C format and TXT format.The information contained in both is consistent, including the model architecture, weights and other necessary information for inference. The C files are convenient for inferencing on the **LC236R5E176_BOARD_V2.0** board.The TXT files are convenient for inferencing on the **EK-U1-VCU128-G** board.

## C files

The C format files include model_data.h and model_data.c.

## TXT files

### Real model data (array). 
    1. data_weight_int8, contains weight data with int8 data type.
    2. data_weight_int32, contains bias data with int32 data type.
    3. data_config, contains all config data, such as stride in conv2d node.
    4. data_quantization_factor, contains quantization factor data, which has been left shifted to a large int32 number. Each factor data is  composed of two value: shifted data and shift number.

### Position data (position means position in real model data array). 
    1. data_average_pool_kernel_shape_pos, contains average_pool kernel shape position in data_config.
    2. data_average_pool_strides_pos, contains average_pool kernel stride position in data_config.
    3. data_conv_activation_func_pos, contains conv2d activation function data position in data_config.
    4. data_conv_bias_pos, contains conv2d bias data position in data_weight_int32.
    5. data_conv_bias_shape_pos, contains conv2d bias shape data position in data_config.
    6. data_conv_pad_pos, contains conv2d pad data position in data_config.
    7. data_conv_quantization_factor_pos, contains conv2d quantization factor data positin in data_quantization_factor.
    8. data_conv_stride_pos, contains conv2d stride data position in data_config.
    9. data_conv_weight_pos, contains conv2d weight data position in data_weight_int8.
    10. data_conv_weight_shape_pos, contains conv2d weight shape data position in data_config.

    11. data_matmul_activation_func_pos, similar to data_conv_activation_func_pos.
    12. data_matmul_bias_pos, similar to data_conv_bias_pos.
    13. data_matmul_bias_shape_pos, similar to data_conv_bias_shape_pos.
    14. data_matmul_quantization_factor_pos, similar to data_conv_quantization_factor_pos.
    15. data_matmul_transa_pos, contains matmul transa data position in data_config.
    16. data_matmul_transb_pos,  contains matmul transb data position in data_config.
    17. data_matmul_weight_pos, similar to data_conv_weight_pos.
    18. data_matmul_weight_shape_pos, similar to data_conv_weight_shape_pos.
    19. data_reshape_shape_pos, contains reshape shape data position in data_config.
    20. data_softmax_axis_pos, contains softmax axis data position in data_config.
    21. data_transpose_perm_pos, contains transpose perm data position in data_config.

### Model architecture data.
    1. data_input2nodes, contains corresponding node_id of input_id.
    2. data_node2inputs, contains corresponding input_id of node_id.
    3. data_node2outputs, contains corresponding output_id of node_id.

