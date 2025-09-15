import argparse

from quantization import *
from utils import *
from graph import *
from serialize import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path', type=str, default='models/vww_96_float.onnx', 
                        help='model file path')
    parser.add_argument('-cp', '--calibration_path', type=str, default='../../data/vww/calibration_official', 
                        help='calibration data path')
    parser.add_argument('-cn', '--calibration_num', type=int, default=200, 
                        help='calibration number')
    parser.add_argument('-ir', '--input_range', type=int, default=1, 
                        help='input range')
    parser.add_argument('-op', '--output_model_path', type=str, default='temps/model_data/', 
                        help='output model data path')
    parser.add_argument('-ic', '--ic_data_file', type=str, default='../../datasets/samples_uint8.pickle', 
                        help='image classification data file')
    parser.add_argument('-mc', '--model_cdata', type=bool, default=False, 
                        help='create model data with format .c and .h')
    parser.add_argument('-mt', '--model_txt', type=bool, default=False, 
                        help='create model data with format .txt')

    args = parser.parse_args()
    onnx_file = args.model_path
    calibration_data_path = args.calibration_path
    calibration_num = args.calibration_num
    input_range = args.input_range
    prefix = args.output_model_path
    ic_data_file = args.ic_data_file
    model_cdata_flag = args.model_cdata
    model_txt_flag = args.model_txt
    
    img_data = None
    if ic_data_file:
        img_data = load_pickle_outputs(ic_data_file)

    graph = extract_onnx(onnx_file)
    activation_factors = inference_calibration_activations(calibration_data_path, img_data, graph, mode="percentile", number=calibration_num, input_range=input_range)

    graph = fuse_activations(graph)
    graph = set_quantization_settings(graph)
    graph = quantize_weights(graph, activation_factors)
    md = ModelData(graph)
    md.create_cdata(prefix, save=model_cdata_flag)
    md.create_txt(prefix, save=model_txt_flag)
