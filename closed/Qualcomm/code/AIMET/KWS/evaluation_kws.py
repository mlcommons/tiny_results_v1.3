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

import get_dataset as kws_data
import kws_util
from AIMET_TestFramework.dataset.tf_processor import Data_Processor
from AIMET_TestFramework.evaluation import load_json
from AIMET_TestFramework.evaluation.model_evaluation import tf_evaluator
from aimet_tensorflow import quantsim
from aimet_tensorflow.cross_layer_equalization import equalize_model
from aimet_tensorflow.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.bias_correction import QuantParams, BiasCorrectionParams, BiasCorrection
import keras_model as models
import eval_functions_eembc as eembc_ev

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


def parser(data, batch, idex, raw_path, input_list, gt_path):
    images = data[0]
    labels = data[1]
    raw_dir_name = os.path.basename(raw_path)
    input_list = open(input_list,'a+')
    input_list.seek(0,2)
    gt = open(gt_path,'a+')
    gt.seek(0,2)
    for i in range(batch):
        raw_name = str(i+idex)+".raw"
        img = images[i].astype(np.float32).tofile(os.path.join(raw_path,raw_name))
        label = (labels[i]).astype(np.int32)
        input_list.write(os.path.join(raw_dir_name,raw_name)+"\n")
        gt.write(str(label)+"\n")
    input_list.close()
    gt.close()

class Data_Processor(Data_Processor):
    def get_tfdataset(self):
        ds_train, ds_test, ds_val = kws_data.get_training_data(self.data_path)
        return ds_test


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
        num_classes = 12
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
        accuracy_eembc = eembc_ev.calculate_accuracy(outputs, labels)
        result = {}
        result["accuracy_eembc"] = str(accuracy_eembc)
        print(result)
        return result


def eval_func(sess, iterations):
    num_classes = 12
    outputs=np.zeros((0,num_classes))
    labels=np.array([])
    idex=0
    
    with sess.graph.as_default():
        dataset = data_processor.get_tfdataset()
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
                #print(out)
                #print(len(out))
                outputs = np.vstack((outputs, out))
                labels = np.hstack((labels, inputs_arr[1]))
                idex+=1
                print(idex)
            # try:
            #     inputs_arr = sess.run(inputs)
            #     input_tensors = [ops + ":0" for ops in model_config["input_node"]]
            #     output_tensors = [ops + ":0" for ops in model_config["output_node"]]
            #     feed_dict = dict(zip(input_tensors, inputs_arr))
            #     out = sess.run(output_tensors,feed_dict=feed_dict)
            #     outputs = np.vstack((outputs, out))
            #     labels = np.hstack((labels, input_tensors[1]))
            #     idex+=1
            #     print(idex)
            except:
                accuracy_eembc = eembc_ev.calculate_accuracy(outputs, labels)
                return accuracy_eembc

        accuracy_eembc = eembc_ev.calculate_accuracy(outputs, labels)
        print("accuracy: " %(str(accuracy_eembc)))
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

Flags, unparsed = kws_util.parse_command()

# converter = tf.lite.TFLiteConverter.from_frozen_graph("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_h5_01.pb", ["input_1"], ["dense/Softmax"], {"input_1":[None,49,10,1]})
# tflite_model = converter.convert()
# open("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_from_pb_api.tflite", "wb").write(tflite_model)
# exit()
# preprocessing = processing.unet_wrap_preprocessing(input_dimension[1],input_dimension[2])
# parser = processing.unet_parser
# data_processor = Data_Processor(data_path, preprocessing, parser)
tf.reset_default_graph()
# sess = tf.Session()
# with tf.gfile.FastGFile("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_h5_01.pb", 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')
# save_tflite(sess, ["input_1"], ["dense/Softmax"], "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_from_pb.tflite")
# exit()

tf.keras.backend.set_learning_phase(0)
sess = tf.Session()
tf.keras.backend.set_session(sess)
model = models.get_model(args=Flags)
model.load_weights("trained_models/kws_model.h5")
sess = tf.keras.backend.get_session()
# freeze_graph(sess,  ["dense/Softmax"], "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_h5_test.pb")
# converter = tf.lite.TFLiteConverter.from_keras_model_file("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model.h5")
# tflite_float_model = converter.convert()
# path = "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_h5_test.tflite"
# open(path, "wb").write(tflite_float_model)
# exit()
# save_tflite(sess, ["input_1"], ["dense/Softmax"], "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model_source_h5_test_save.tflite")
# exit()
### 
# model = tf.keras.models.load_model("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model.h5")


initialize_uninitialized_vars(sess)
if eval_source:
    data_processor = Data_Processor(Flags, None, parser)
    data_processor.batch = Flags.batch_size

    model_config = "model_info.json"
    model_config = load_json.load_json(model_config)
    evaluator = tf_evaluator(sess,sess,model_config,data_processor)
    evaluator.set_accuracy_algo(accuracy_md)
    results = evaluator.run()
    exit()
with sess.as_default():
# saver = tf.train.Saver()
# saver.save(sess, "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/kws_ckpt/kws.ckpt")

# from tensorflow.python.platform import gfile
# saver=tf.train.import_meta_graph('/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/kws_ckpt/kws.ckpt.meta')
# saver.restore(sess, tf.train.latest_checkpoint('/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/kws_ckpt'))
    data_processor = Data_Processor(Flags, None, parser)
    data_processor.batch = Flags.batch_size
    dataset = data_processor.get_tfdataset()
    iterator = dataset.make_initializable_iterator()

    model_config = "model_info.json"
    model_config = load_json.load_json(model_config)
    quantsim_config_file = "config_for_eai.json"
    sess.run(iterator.initializer)

    # with open("model_nodes",'w') as f:
    #     graph = tf.get_default_graph()
    #     [f.write(str(n)) for n in tf.get_default_graph().as_graph_def().node]
    #########
    #freeze_graph(sess,"/soft/Myscripts/tiny_ml/01")
    eval_func(sess,-1)
    #graph_def = tf.graph_util.remove_training_nodes(sess.graph_def)
    input_op_name = model_config["input_node"]
    output_op_name =model_config["output_node"]

### fold batch norm
    sess, folded_pairs = fold_all_batch_norms(sess, input_op_name, output_op_name)
    """
### CLE
    # new_session = sess
    new_session = equalize_model(sess, input_op_name[0], output_op_name[0])
### BC
    quant_params = QuantParams(quant_mode='tf_enhanced',
                                round_mode='nearest',
                                use_cuda=True,
                                ops_to_ignore=None)
    bias_correction_params = BiasCorrectionParams(batch_size=Flags.batch_size,
                                                  num_quant_samples=10,
                                                  num_bias_correct_samples=10,
                                                  input_op_names=input_op_name,
                                                  output_op_names=output_op_name)
    # run bias correction on the model
    def convert_for_bc(i_0,i_1):
        return i_0
    dataset = data_processor.get_tfdataset()
    iterator = dataset.make_initializable_iterator()
    dataset_bc = dataset.map(convert_for_bc)
    new_session = BiasCorrection.correct_bias(new_session, bias_correction_params, quant_params, dataset_bc)
    """
# with new_session.graph.as_default():
#     saver = tf.train.Saver()
#     saver.save(new_session, "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/bc_model/bc_output.ckpt")

# tf.reset_default_graph()
# sess = tf.compat.v1.Session(graph=tf.Graph())
# with sess.graph.as_default():
#     saver = tf.compat.v1.train.import_meta_graph("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/bc_model/bc_output.ckpt.meta")
#     saver.restore(sess, "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/bc_model/bc_output.ckpt")

# data_processor = Data_Processor(Flags, None, parser)
# data_processor.batch = Flags.batch_size
new_session = sess 
# eval_func(new_session,-1)
sim = quantsim.QuantizationSimModel(new_session,
                                    starting_op_names=input_op_name,
                                    output_op_names=output_op_name,
                                    quant_scheme='tf_enhanced',
                                    default_output_bw=8,
                                    default_param_bw=8,
                                    config_file=quantsim_config_file)   
sim.compute_encodings(eval_func, forward_pass_callback_args=120)
# eval_func(sim.session,-1)
# save_tflite(sim.session, ["input_1"], ["dense/Softmax"], "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model__cle_bc.tflite")
# exit()



sess = tf.Session()
tf.keras.backend.set_session(sess)
model = models.get_model(args=Flags)
model.load_weights("trained_models/kws_model.h5")

### 
# model = tf.keras.models.load_model("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model.h5")
tf.keras.backend.set_learning_phase(0)
sess = tf.keras.backend.get_session()
initialize_uninitialized_vars(sess)
data_processor = Data_Processor(Flags, None, parser)
data_processor.batch = Flags.batch_size

evaluator = tf_evaluator(sess,sim,model_config,data_processor)
evaluator.set_accuracy_algo(accuracy_md)
results = evaluator.run()
# saver = tf.train.Saver()
# saver.save(sess, "tested_model/ckpt")


