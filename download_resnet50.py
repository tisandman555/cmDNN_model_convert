from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from keras import backend as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

#download the model from keras model zoo and convert it to tensorflow format
#model url https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
model = ResNet50(weights='imagenet')

output_model_name = "Resnet50.pb"
output_model_dir = "tf_model"

K.set_learning_phase(0)
sess = K.get_session()


orig_output_node_names = [node.op.name for node in model.outputs]

constant_graph = graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    orig_output_node_names)
graph_io.write_graph(
    constant_graph,
    output_model_dir,
    output_model_name,
    as_text=False)


print('ok')
