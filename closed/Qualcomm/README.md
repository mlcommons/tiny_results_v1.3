# Qualcomm MLPerf™ Tiny benchmark v1.3
## Preparation
DUT: Next_Generation_Snapdragon_Mobile_Platform_MTP<br>
Meta: Kaanapali.LA.1.0.r1-00850-STD.INT-1<br>
eAI SDK: eAI release 5.9<br>
Hexagon SDK: Hexagon SDK 6.2.0.1<br>
<br>
>### Flash meta build 
```bash
adb reboot fastboot
python ${META_PATH}/common/build/fastboot_complete.py
```
### push sysmon to device
You can find sysmon app under ADSP image folder. e.g. LPAIDSP.HT.1.2-00749.1-KAANAPALI-1
```bash
adb push ${ADSP_PATH}/adsp_proc/performance/sysmonapp/sysMonApp /data/local/tmp/
adb shell chmod 777 /data/local/tmp/sysMonApp
```
>### Set up eAI SDK environment in Linux
To set up SDK environment in Linux, you must first to switch to a bash shell. To switch from any unknown shell to a bash shell in Linux, enter bash in the terminal. This step is required because the setup script works in the bash environment.<br>
Make sure you can find conda in your PATH and create a new conda environment (using the eai.yml):
```bash
conda env create -f code/eai.yml
conda activate eai
# Install the C++ Multilib
sudo apt-get install g++-multilib
```
Hexagon SDK is necessary for eAI build environement. Please install the Hexagon SDK locally and set up build environment.<br>
```bash
source ${HEXAGON_SDK}/setup_sdk_env.source
```
>### Testsig on device
```bash
adb root
adb wait-for-device
# adb disable-verity may be needed if adb remount fails with permission denied.
# After that, do adb reboot, and then try adb root, and adb remount
# DO NOT adb enable-verity after - otherwise device will go into boot loop
adb remount

cd ${HEXAGON_SDK}/tools/elfsigner
#(e.g. cd /root/hexagon-sdk-5.5.0.1/tools/elfsigner)
adb push getserial/ADSP/android_Release/getserial /data/local/tmp
#! <ADSP> is new layer add getserial
#(optional: adb shell chmod 777 /data/local/tmp/getserial)
adb shell /data/local/tmp/getserial
#** LOOK at the output.  It will give you something like "Serial Num : 0xdeadbeef ***
 
#alternatively if getserial doens't work for some reason, you can use
adb shell cat /sys/devices/soc0/serial_number
#Look at the output. It will give you a decimal number like 3735928559 (in this example, its the decimal value of the hexadecimal 0xdeadbeef)
  
# Ensure to run elfsigner.py in python2.7 environment [seems to have import issues running with python3]
# also make sure to use the hex serial number including the 0x prefix
python elfsigner.py -t <serial number from above>
#** when asked to Agree, press "y"
#** Notice that it now provided a full path to testsig-<serial #>.so like so:
#**** Signing complete! Output saved at ${HEXAGON_SDK}/tools/elfsigner/output/testsig-0x2ac6fac3.so
adb push <fullpathtothetestsig> /vendor/lib/rfsa/adsp/  #(OR /system/vendor/lib/rfsa/adsp, or /usr/lib/rfsa/adsp for Ubuntu ARM)
adb reboot
```
## Model conversion
The KWS, IC and VWW models were symmetrically quantified by AIMET.<br>
AIMET quantization scripts are as follows and the code to generate the tflite model is also included in the following script:<br>
IC: code/AIMET/IC/evaluation_imgclass_benchmark.py<br>
KWS: code/AIMET/KWS/evaluation_kws_benchmark.py<br>
VWW: code/AIMET/VWW/evaluation_vww.py<br>
Please use the tool “eai_builder” in eAI SDK to convert the models. <br>
>### Conversion Commands:
```bash
${EAI_SDK}/eai_runtime/tools/model_builder/eai_builder --tflite ad01_int8.tflite --enable_enpu_ver v6 --enable_layer_fusion 1 --enable_channel_align 1 --force_nhwc_layout 1 --output ad_model.eai
${EAI_SDK}/eai_runtime/tools/model_builder/eai_builder --tflite kws_saved_model.tflite --quantization_config_file kws_aimet_symmetric.json --enable_enpu_ver v6 --enable_layer_fusion 1 --enable_channel_align 1 --output kws_model.eai
${EAI_SDK}/eai_runtime/tools/model_builder/eai_builder --tflite vww_keras_model.tflite --quantization_config_file vww_aimet_symmetric.json --enable_enpu_ver v6 --enable_layer_fusion 1 --enable_channel_align 1 --force_nhwc_layout 1 --output vww_model.eai
${EAI_SDK}/eai_runtime/tools/model_builder/eai_builder --tflite pretrainedResnet.tflite --quantization_config_file ic_aimet_symmetric.json --enable_enpu_ver v6 --enable_layer_fusion 1 --enable_channel_align 1 --force_nhwc_layout 1 --output ic_model.eai
```
After the model conversion is successful, you will get the following 4 models under the current working director，please do not rename them.<br>
```bash
tree $PWD
$PWD
├── ad_model.eai
├── ic_model.eai
├── kws_model.eai
└── vww_model.eai
```
>### Note
For the image classification model, to achieve better accuracy, we used open sourced [AIMET](https://github.com/quic/aimet) tool to generate the quantization config file (code/image_classification/ic_aimet_symmetric.json) for eAI model builder. 

## Build libs
>### Copy source code to eAI SDK
```bash
cp -r code/anomaly_detection ${EAI_SDK}/example/
cp -r code/image_classification ${EAI_SDK}/example/
cp -r code/keyword_spotting ${EAI_SDK}/example/
cp -r code/person_detection ${EAI_SDK}/example/
```
>### Copy CMakeLists to eAI SDK
```bash
cp code/CMakeLists.txt ${EAI_SDK}/
```
>### build binaries
```bash
cd ${EAI_SDK}
./build.sh --build_adsp --clean_build --build_enpu_ver 6
```
If the build is successful you can find the generated files in the ${EAI_SDK}/build-fixed32/example.
## Push resources
Connect devices with USB Type-C
```bash
adb root
adb wait-for-device remount
# push test resources to DUT
adb push ad_model.eai /data/local/tmp/
adb push ic_model.eai /data/local/tmp/
adb push kws_model.eai /data/local/tmp/
adb push vww_model.eai /data/local/tmp/
adb push ${EAI_SDK}/build-fixed32/example/anomaly_detection/libeai_tiny_ad.so /vendor/lib/rfsa/adsp/
adb push ${EAI_SDK}/build-fixed32/example/image_classification/libeai_tiny_ic.so /vendor/lib/rfsa/adsp/
adb push ${EAI_SDK}/build-fixed32/example/keyword_spotting/libeai_tiny_kws.so /vendor/lib/rfsa/adsp/
adb push ${EAI_SDK}/build-fixed32/example/person_detection/libeai_tiny_vww.so /vendor/lib/rfsa/adsp/
adb push ${EAI_SDK}/binaries/enpu_launcher/enpu_launcher /vendor/bin/
adb push ${EAI_SDK}/binaries/enpu_launcher/libenpu_launcher_skel.so /vendor/lib/rfsa/adsp/
```
## Launch test app
Connect DUT with UART port with UART communication tool(e.g. using [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)).UART config:
> baudRate: 115200<br>
> dataBits: 8<br>
> parity: None<br>
> stopBits: 1<br>

After successfully connecting with the DUT, please send the following commands through UART
```bash
su
stty raw
cd /data/local/tmp
setprop vendor.fastrpc.process.rpctimeout 0
export ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp
/data/local/tmp/sysMonApp clocks set --coreClock 3000 --busClock 3000 --q6 adsp
#Launch the test app for specified model. e.g. anomaly_detection: 
enpu_launcher libeai_tiny_ad.so 
```
## Start the runner test
### Test with EEMBC runner
Close the UART connection and launch [EEMBC runner](https://github.com/eembc/energyrunner/) app. Please follow the [EEMBC runner](https://github.com/eembc/energyrunner/) instructions for testing process.
### Test with MLC runner
Tools can be found in TinyML repo: [MLC runner](https://github.com/mlcommons/tiny/tree/master/benchmark/runner)
Close the UART connection and run the script:
```bash
python main.py --dataset_path=runner\benchmarks\ulp-mlperf\datasets --test_script=tests_performance.yaml --device_list=devices_kws_ic_vww.yaml --mode=p
python main.py --dataset_path=runner\benchmarks\ulp-mlperf\datasets --test_script=tests_accuracy.yaml --device_list=devices_kws_ic_vww.yaml --mode=a
```
## Close the thread of test app 
```bash
# send the following command to stop the test app
%%
# ifstart testing other models, launch the test app for specified model.
# CMD: enpu_launcher libeai_tiny_${MODEL}.so 
# ${MODEL} must be one of {vww, ic, kws, ad}
```
## The results
The logs of performance and accuracy can be found in the results folder. The are the outputs if EEMBC runner. 
|model|accuracy|latency in ms|
|:-----:|:----:|:----:|
| AD | AUC: 0.86 | 0.068 |
| IC | Top-1: 85.5% | 0.096 |
| KWS | Top-1: 91.3% | 0.065 |
| VWW | Top-1: 83.5% | 0.114 |

