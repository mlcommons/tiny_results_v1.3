# MLPerf Tiny benchmark v1.3 - Closed Division

This document provides an overview of our submission to the MLPerf Tiny benchmark v1.3. The benchmarks were recorded on the **EK-U1-VCU128-G** and **K1_BOARD_V1.0**. We run the **vww** benchmark using both **EK-U1-VCU128-G** and **K1_BOARD_V1.0**, and run the **ic**, **kws**, **ad** benchmark only using **K1_BOARD_V1.0**. The submission contains performance results. Our solutions are as below: 

## K1_BOARD_V1.0

### Model conversion

In our solution, we use the format of ONNX models for inference, while the models provided are the format of tflite, we convert the original models firstly.

For the models of ic, vww and ad, the tflite models can be converted to the format of ONNX using the python scripts which can be found in ```code/ic/K1_BOARD_V1.0/model_conversion```. For the models of kws, the tflite models can be converted using the tf2onnx tool by the shell scripts in ```code/ic/K1_BOARD_V1.0/model_conversion```. Make sure you have installed all the python packages needed.

### Model quantization

Use our model quantizing tool for model quantization. The quantizing tool supports the format of **jpg** and **bin** quantified calibration data file. In this solution, we use the jpg data file for the quantization of vww model, and bin data file for the quantization of ad, ic and kws models. For the **ic** and **vww** benchmark, the quantization calibration data can be found in ```code/{model_name}/K1_BOARD_V1.0/calib_data``` where ```model_name``` here represent the name of each model. For the **kws** and **ad** benchmark, the quantization calibration data can be created by the shell script inside the ```code/{model_name}/K1_BOARD_V1.0/calib_data``` path. The wheel package of quantizing tool can be found in ```code/ic/K1_BOARD_V1.0/quantization_tool```. Install the quantizing tool and then quantize the model by the command ```python3 -m xquant --config <path_to_json_file>```, where ```path_to_json_file``` is the path of configure json files for each model, which can be found in ```code/ic/K1_BOARD_V1.0/quantization_tool```. The final quantized model are placed in the ```code/{model_name}/K1_BOARD_V1.0/models```.

### Adding UART device to the board

The UART device can be added to the linux system using the **device tree blob**, replace the old dtb by the dtb file placed in ```code/ic/K1_BOARD_V1.0/dtb```, reboot the board and verify whether ```/dev/ttyS7``` exists.

### Benchmark deployment

In the **ic** benchmark, the binary file is the ```code/ic/K1_BOARD_V1.0/resnet```.  Put the binary file into the linux system inside K1 processor, create a folder named ```models``` in the same path, and then copy the quantized model file to the folder.

In the **vww**, **ad**, **kws** benchmark, the binary file is the  ```code/vww/K1_BOARD_V1.0/mobilenet```, ```code/ad/K1_BOARD_V1.0/deep_autoencoder```, ```code/kws/K1_BOARD_V1.0/dscnn``` respectively. And the way to deploy is the same as **ic** benchmark.

### Prepare for testing

1. Connect the TXD of the USB-to-serial adapter to pin24 of the board's J24 header, RXD to pin23 of J24, and GND to pin25 of J24. Plug the USB end into the host computer.
2. Power on the board and execute the corresponding framework binary program with root privileges on the board side.

## EK-U1-VCU128-G

### Generate quantized model data

Use our model parsing tool for model parsing, model quantization, and model serialization to generate quantized model data files. The model parsing tool is placed in ```code/vww/VCU128/graph_quantization```. These model data files are placed in ```code/vww/VCU128/graph_quantization/temps```, including model architecture, weight data, quantization information, and other data.

### Generate ANPU hardware execution control instructions

Based on the model architecture data, we dissect the model manually and generate **ANPU** hardware execution control instructions, which is located at ```code/vww/VCU128/hardware/Hardware_Exucution_Control_Instruction_for_ANPU.xlsx```. 

### Generate model data used for ANPU

Use scripts to rearrange weight data, quantization data, and quantization offset data in the model data files to generate weight data, quantization data, and quantization offset data used for **ANPU**. These scripts are placed in ```code/vww/VCU128/hardware/convert_to_anpu_model_data```, and their outputs are placed in ```code/vww/VCU128/hardware/convert_to_anpu_model_data/anpu_model_data```.

### Integrate into an executable software program

Integrate **ANPU**'s hardware execution control instructions, weight data, quantization data, and quantization offset data into an executable software program.

### Prepare for testing

1. Power on the **EK-U1-VCU128-G** board; use a **JTAG** downloader to write the **SoC+ANPU** BIT file, which is located at ```code/vww/VCU128/hardware/hw.bit```. Then check if the **SoC** is working properly using **CDK** IDE, and flash the executable software program onto the **SoC**. The software program bit file is located at ```code/vww/VCU128/sw.elf```. Resources occupied by hardware_design is shown in ```code/vww/VCU128/hardware/Resources_Occupied_By_Hardware_Design.png```
2. The executable software program sends **ANPU**'s hardware execution control instructions, weight data, quantization data, and quantization offset data to **ANPU**.
3. The host reads an image from the dataset and sends it to the **SoC** via a serial port. When the **SoC** receives it, it sends the image to **ANPU** through **DMA**. **ANPU** caches the image and all intermediate calculation results in the accelerator memory and sends the calculation results to the **SoC** after the computation is done.
4. The **SoC** performs data post-processing and sends it to the host via a serial port, completing the computation process for a single image.