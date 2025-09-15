# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:37:31 2021

@author: dyou
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import argparse
#import keras
from sklearn.metrics import roc_auc_score

import train
import eval_functions_eembc
import keras_model
import json
import logging
from aimet_tensorflow import quantsim
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.bias_correction import QuantParams, BiasCorrectionParams, BiasCorrection
import os


model_name = keras_model.get_model_name()

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

def keras_model_to_tflite(model, path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(path, "wb").write(tflite_model)

def get_quantization_dataset(batch):
    cifar_10_dir = 'cifar-10-batches-py'
    sample_img = []
    sample_labels = []
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)
    print(test_data.shape)
    _idx = np.load(cal_file)
    for i in _idx:
        sample_img.append(np.array(test_data[i], dtype=np.float32))
        sample_labels.append(np.array(test_labels[i], dtype=np.float32))
    return tf.data.Dataset.from_tensor_slices((np.array(sample_img),np.array(sample_labels))).repeat(1).apply(tf.contrib.data.batch_and_drop_remainder(batch))

def eval_quant_func(sess, iterations):
        num_classes = 10
        outputs=np.zeros((0,num_classes))
        labels=np.array([])
        idex=0
        with sess.graph.as_default():
            dataset = get_quantization_dataset(batch=1)
            iterator = dataset.make_initializable_iterator()
            inputs = iterator.get_next()
            sess.run(iterator.initializer)
            while idex < iterations or iterations == -1:
                try:
                    inputs_arr = sess.run(inputs)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--cal_file',
    type=str,
    default="calibration_samples_idxs.npy",
    help="calibration data index")
    parser.add_argument(
    '--model_info',
    type=str,
    default="model_info.json",
    help="model infomation")
    parser.add_argument(
    '--quantsim_config',
    type=str,
    default="htp_quantsim_config.json",
    help="quantization simulation config")
    parser.add_argument(
    '--params',
    type=str,
    default="trained_models/pretrainedResnet.h5",
    help="model parameter")
    args=parser.parse_args()

    cal_file = args.cal_file
    model_config = args.model_info
    quantsim_config_file = args.quantsim_config
    h5_model = args.params
    
    cifar_10_dir = 'cifar-10-batches-py'
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)
    label_classes = np.argmax(test_labels,axis=1)
    print("Label classes: ", label_classes.shape)

    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(0)
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    model = keras_model.resnet_v1_eembc()
    model.load_weights(h5_model)

    model_config = load_json(model_config)
    input_op_name = model_config["input_node"]
    output_op_name =model_config["output_node"]

    sess, folded_pairs = fold_all_batch_norms(sess, input_op_name, output_op_name)
    save_tflite(sess, input_op_name, output_op_name, "source_model.tflite")    
    
    sim = quantsim.QuantizationSimModel(sess,
                                    starting_op_names=input_op_name,
                                    output_op_names=output_op_name,
                                    quant_scheme='tf',
                                    default_output_bw=8,
                                    default_param_bw=8,
                                    config_file=quantsim_config_file)
    sim.compute_encodings(eval_quant_func, forward_pass_callback_args=500)
    sim.export("results","ic_tf_asym_w8_a8")

    eval_quant_func(sim.session, -1)

    # save_tflite(sim.session, input_op_name, output_op_name, "source_model_aimet.tflite")   

