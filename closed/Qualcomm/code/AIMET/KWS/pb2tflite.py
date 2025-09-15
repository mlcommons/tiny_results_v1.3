# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:37:31 2021

@author: dyou
"""
import tensorflow as tf

sess = tf.Session()
with tf.gfile.FastGFile("/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/kws_aimet.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

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

save_tflite(sess, ["input_1"], ["dense/Softmax"], "/soft/Myscripts/tiny_ml/tiny/benchmark/training/keyword_spotting/trained_models/kws_model__cle_bc.tflite")

