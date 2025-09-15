from tensorflow.keras.models import load_model
import onnx
import tf2onnx
import tensorflow as tf

# Load .h5 model
model = load_model('./models/pretrainedResnet.h5')

# Set input data shape
input_signature = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input_1"),)

# Convert to the model of ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

# Save ONNX model
with open('./models/my_resnet_test.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())
