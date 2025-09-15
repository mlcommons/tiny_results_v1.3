# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:37:31 2021

@author: dyou
"""
import tensorflow as tf
import os
import numpy as np
import argparse
import json
# import get_dataset as kws_data
import kws_util
import logging 
from aimet_tensorflow import quantsim
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.bias_correction import QuantParams, BiasCorrectionParams, BiasCorrection
import keras_model as models
import eval_functions_eembc as eembc_ev

Flags, unparsed = kws_util.parse_command()
Flags.batch_size = 1
eval_source = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
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

def initialize_uninitialized_vars(sess):
    """
    Some graphs have variables created after training that need to be initialized.
    However, in pre-trained graphs we don't want to reinitialize variables that are already
    which would overwrite the values obtained during training. Therefore search for all
    uninitialized variables and initialize ONLY those variables.
    :param sess: TF session
    :return:
    """
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) for var in global_vars])
    uninitialized_vars = list(compress(global_vars, is_not_initialized))
    if uninitialized_vars:
        sess.run(tf.variables_initializer(uninitialized_vars))

def get_tfdataset(Flags):
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    return ds_val

def eval_func(sess, iterations):
    num_classes = 12
    outputs=np.zeros((0,num_classes))
    labels=np.array([])
    idex = 0
    nr = -1 # 0
    print(cal_indices)
    with sess.graph.as_default():
        dataset = get_tfdataset(Flags)
        iterator = dataset.make_initializable_iterator()
        inputs = iterator.get_next()
        sess.run(iterator.initializer)
        while idex < iterations or iterations == -1:
            try:
                inputs_arr = sess.run(inputs)
                nr += 1
                if nr not in cal_indices:
                    continue
                input_tensors = [ops + ":0" for ops in model_config["input_node"]]
                output_tensors = [ops + ":0" for ops in model_config["output_node"]]
                feed_dict = dict(zip(input_tensors, inputs_arr))
                out = sess.run(output_tensors[0],feed_dict=feed_dict)
                outputs = np.vstack((outputs, out))
                labels = np.hstack((labels, inputs_arr[1]))
                idex+=1
                print(idex)
            except:
                accuracy_eembc = eembc_ev.calculate_accuracy(outputs, labels)
                print("accuracy on valset: %s" %(str(accuracy_eembc)))
                return accuracy_eembc

        accuracy_eembc = eembc_ev.calculate_accuracy(outputs, labels)
        print("accuracy on valset: %s" %(str(accuracy_eembc)))
        return accuracy_eembc

def freeze_graph(sess, outputs, path):
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, outputs
    )
    with tf.gfile.FastGFile(path, mode="wb") as f:
        f.write(constant_graph.SerializeToString())

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--cal_file',
    type=str,
    default="quant_cal_idxs.txt",
    help="calibration dataset")
    parser.add_argument(
    '--model_info',
    type=str,
    default="model_info.json",
    help="model infomation")
    parser.add_argument(
    '--quantsim_config',
    type=str,
    default="config_for_eai.json",
    help="quantization simulation config")
    parser.add_argument(
    '--params',
    type=str,
    default="trained_models/kws_ref_model/variables",
    help="model parameter")
    args=parser.parse_args()
    # args=parser.__dict__
    calibration_file = args.cal_file
    model_config = args.model_info
    quantsim_config_file = args.quantsim_config
    weights = args.params
    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(0)
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    model = tf.keras.models.load_model("trained_models/kws_ref_model")
    model.save_weights("origin_saved_model_to_keras")
    model = models.get_model(args=Flags)
    model.load_weights("origin_saved_model_to_keras")
    model.save("origin_saved_model_to_keras_h5")

    with open(calibration_file) as fpi:
        cal_indices = [int(line) for line in fpi]
    cal_indices.sort()

    initialize_uninitialized_vars(sess)
    model_config = load_json(model_config)
    
    with open("model_nodes",'w') as f:
        graph = tf.get_default_graph()
        [f.write(str(n)) for n in tf.get_default_graph().as_graph_def().node]

    input_op_name = model_config["input_node"]
    output_op_name =model_config["output_node"]

    save_tflite(sess, input_op_name, output_op_name, "kws_saved_model.tflite")

    ### fold batch norm
    sess, folded_pairs = fold_all_batch_norms(sess, input_op_name, output_op_name)

    sim = quantsim.QuantizationSimModel(sess,
                                        starting_op_names=input_op_name,
                                        output_op_names=output_op_name,
                                        quant_scheme='tf',
                                        default_output_bw=8,
                                        default_param_bw=8,
                                        config_file=quantsim_config_file)   
    sim.set_and_freeze_param_encodings("freez_input_encodings.json")

    sim.compute_encodings(eval_func, forward_pass_callback_args=-1)
    sim.export("results", "kws")
