# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:37:31 2021

@author: dyou
"""
import tensorflow as tf
import os
import numpy as np
import argparse
import tensorflow.keras.backend as K
import datetime
import json

from aimet_tensorflow import quantsim
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.bias_correction import QuantParams, BiasCorrectionParams, BiasCorrection
import eval_functions_eembc
from vww_model import mobilenet_v1
logger = logging.getLogger(__name__)
def load_json(json_path):
    try:
        with open(json_path, "r") as json_file:
            try:
                json_data = json.load(json_file)
                return json_data
            except ValueError as e:
                logger.error("error parsing JSON file: " + json_path + " " + repr(e))
                return []
    except Exception as e:
        logger.error("error opening file: " + json_path + " " + repr(e))

class accuracy_md(object):
    def __init__(self, *args, **kwargs):
        try:
            self.logger = kwargs["logger"]
            self.output_raw = kwargs["output_raw"]
            self.output_dir = kwargs["output_dir"]
            self.input_list = kwargs["input_list"]
            self.y_true_file = kwargs["ground_truth"]
            self.labels_file = kwargs["labels"]
        except KeyError as e:
            self.logger.error(e)
        self.logger.info("init accuracy egine: %s" % self.__class__.__name__)
    def get_output_length(self):
        output_dir_list = os.listdir(self.output_dir)
        result_list = [x for x in output_dir_list if "Result_" in x]
        return len(result_list)
    def get_cur_result_dir(self):
        return "Result_{idx}"
    def get_acc(self):
        num_classes = 2
        labels=np.array([])
        outputs=np.zeros((0,num_classes))
        with open(self.y_true_file, "r") as f:
            for line in f.readlines():
                labels = np.hstack((labels, int(line.strip())))
        with open(self.input_list, "r") as f:
            input_files = [line.strip().split("/")[-1] for line in f.readlines()]
            input_length = len(input_files)
            #assert input_length == self.get_output_length()
        for idx, val in enumerate(input_files):
            cur_results_dir = self.get_cur_result_dir().replace("{idx}", str(idx))
            cur_results_file = os.path.join(
                    self.output_dir, cur_results_dir, self.output_raw
            )
            float_array = np.fromfile(cur_results_file, dtype=np.float32)
            float_array = np.reshape(float_array,(1,num_classes))
            outputs =  np.vstack((outputs, float_array))
        accuracy_eembc = eval_functions_eembc.calculate_accuracy(outputs, labels)
        result = {}
        result["accuracy_eembc"] = str(accuracy_eembc)
        print(result)
        return result

def save_tflite(sess, inputs, outputs, path):
    g = sess.graph
    with g.as_default():
        in_tensor = []
        out_tensor = []
        for i in inputs:
            in_tensor.append(g.get_tensor_by_name(name='%s:0'%i))
        for o in outputs:
            out_tensor.append(g.get_tensor_by_name(name='%s:0'%o))

        converter = tf.lite.TFLiteConverter.from_session(sess, in_tensor, out_tensor)
        tflite_model = converter.convert()
        open(path, "wb").write(tflite_model)
    
class Data_Processor(object):
    def __init__(self) -> None:
        super().__init__()
    def get_tfdataset(self):
        energy_data_path = "energyrunner-3.0.6/datasets/vww01"
        y_labels_path = "energyrunner-3.0.6/datasets/vww01/y_labels.csv"
        data = []
        labels = []
        with open(y_labels_path) as f:
            for line in f.readlines():
                label_tmp= np.zeros(2)
                name,_,label = line.split(",")
                data_tmp = np.fromfile(os.path.join(energy_data_path,name), dtype = np.uint8)
                # data_tmp = data_tmp + 128
                data_tmp = data_tmp/255.
                data_tmp = data_tmp.reshape((96,96,3))
                data.append(data_tmp)
                label_tmp[int(label)] = 1
                labels.append(label_tmp)
        data = np.array(data)
        labels = np.array(labels)
        print(data.shape)
        print(labels.shape)
        return tf.data.Dataset.from_tensor_slices((data,labels)).repeat(1).apply(tf.contrib.data.batch_and_drop_remainder(self.batch))

    def get_quantdataset(self):
        calibration_path = "calibration_data.txt"
        vww_dir = "vw_coco2014_96"
        full_list = []
        for dir in os.listdir(vww_dir):
            target_dir = os.path.join(vww_dir,dir)
            for file in os.listdir(target_dir):
                file = os.path.join(target_dir,file)
                full_list.append(file)
        cal_list = []
        with open(calibration_path) as f:
            for line in f.readlines():
                line = line.strip()
                for target_file_path in full_list:
                    if line in target_file_path:
                        cal_list.append(target_file_path)
        output_array = []
        output_label = []
        for cal_file in cal_list:
            if "non_persor" in cal_file:
                label = [1,0]
            else:
                label = [0,1]
            img = tf.keras.preprocessing.image.load_img(
                    cal_file, color_mode='rgb').resize((96, 96))
            arr = tf.keras.preprocessing.image.img_to_array(img)
            output_array.append(arr.reshape(96, 96, 3) / 255.)
            output_label.append(label)
            
        output_array = np.array(output_array)
        output_label = np.array(output_label)
        return tf.data.Dataset.from_tensor_slices((output_array,output_label)).repeat(1).apply(tf.contrib.data.batch_and_drop_remainder(self.batch))

def parser(data, batch, idex, raw_path, input_list, gt_path):
    images = data[0]
    labels = data[1]
    labels = np.argmax(data[1],axis=1)
    raw_dir_name = os.path.basename(raw_path)
    input_list = open(input_list,'a+')
    input_list.seek(0,2)
    gt = open(gt_path,'a+')
    gt.seek(0,2)
    for i in range(batch):
        raw_name = str(i+idex)+".raw"
        images[i].astype(np.float32).tofile(os.path.join(raw_path,raw_name))
        label = (labels[i]).astype(np.int32)
        input_list.write(os.path.join(raw_dir_name,raw_name)+"\n")
        gt.write(str(label)+"\n")
    input_list.close()
    gt.close()

def freeze_ckpt(meta, outputs, path):
    # tf.reset_default_graph()
    tmp_graph = tf.Graph()
    with tmp_graph.as_default():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(meta)
        saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(meta)))
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, outputs
        )
        save_tflite(sess, input_op_name, output_op_name, path)
        # with tf.gfile.FastGFile(path, mode="wb") as f:
        #     f.write(constant_graph.SerializeToString())

if __name__ == "__main__":
    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(0)
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    model_path = "trained_models/vww_96.h5"
    model = mobilenet_v1()
    model.load_weights(model_path)

    model_config = "model_info.json"
    model_config = load_json.load_json(model_config)
    input_op_name = model_config["input_node"]
    output_op_name =model_config["output_node"]

    save_tflite(sess, input_op_name, output_op_name, "source.tflite")

    data_processor = Data_Processor(None, None, parser)
    data_processor.batch = 1
    
    def eval_func(sess, iterations):
            num_classes = 2
            outputs=np.zeros((0,num_classes))
            labels=np.array([])
            idex=0
            nr = -1 # 0
            with sess.graph.as_default():
                dataset = data_processor.get_quantdataset()
                iterator = dataset.make_initializable_iterator()
                inputs = iterator.get_next()
                sess.run(iterator.initializer)
                while idex < iterations or iterations == -1:
                    try:
                        inputs_arr = sess.run(inputs)
                        nr += 1
                        # if nr not in cal_indices:
                        #     continue
                        input_tensors = [ops + ":0" for ops in model_config["input_node"]]
                        output_tensors = [ops + ":0" for ops in model_config["output_node"]]
                        feed_dict = dict(zip(input_tensors, inputs_arr))
                        out = sess.run(output_tensors[0],feed_dict=feed_dict)
                        outputs = np.vstack((outputs, out))
                        labels = np.hstack((labels, np.argmax(inputs_arr[1],axis=1)))
                        idex+=1
                    except:
                        accuracy_eembc = eval_functions_eembc.calculate_accuracy(outputs, labels)
                        print("accuracy on energy: %s" %(str(accuracy_eembc)))
                        return accuracy_eembc

                accuracy_eembc = eval_functions_eembc.calculate_accuracy(outputs, labels)
                print("accuracy on energy: %s" %(str(accuracy_eembc)))
                return accuracy_eembc
    #eval_func(sess,-1)
    sess, folded_pairs = fold_all_batch_norms(sess, input_op_name, output_op_name)
    quantsim_config_file = "config_for_eai.json"

    # # CLE
    new_session = equalize_model(sess, input_op_name[0], output_op_name[0])
    # # BC
    quant_params = QuantParams(quant_mode='tf',
                                round_mode='nearest',
                                use_cuda=False,
                                ops_to_ignore=None)
    bias_correction_params = BiasCorrectionParams(batch_size=1,
                                                  num_quant_samples=12,
                                                  num_bias_correct_samples=12,
                                                  input_op_names=input_op_name,
                                                  output_op_names=output_op_name)
    # run bias correction on the model
    def convert_for_bc(i_0,i_1):
        return i_0
    dataset = data_processor.get_quantdataset()
    iterator = dataset.make_initializable_iterator()
    dataset_bc = dataset.map(convert_for_bc)
    new_session = BiasCorrection.correct_bias(new_session, bias_correction_params, quant_params, dataset_bc)



    sim = quantsim.QuantizationSimModel(new_session,
                                    starting_op_names=input_op_name,
                                    output_op_names=output_op_name,
                                    quant_scheme='tf',
                                    default_output_bw=8,
                                    default_param_bw=8,
                                    config_file=quantsim_config_file)
    #sim.session.dense.Softmax.output_quantizer.enabled = True
    sim.compute_encodings(eval_func, forward_pass_callback_args=12)
    export_name = "vww_symmetric_w8_a8_cle_bc"
    sim.export("results",export_name)
    # exit()
    eval_func(sim.session, -1)
    freeze_ckpt(f"results/{export_name}.meta",output_op_name,f"results/{export_name}_aimet_changed_cle_bc.tflite")

