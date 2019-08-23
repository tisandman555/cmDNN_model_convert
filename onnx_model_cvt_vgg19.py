import sys
import numpy as np
import struct
sys.path.append('C:\\temp_20151027\\model_optimizer')

import onnx
from onnx import numpy_helper

#the model is downloaded from https://s3.amazonaws.com/download.onnx/models/opset_8/vgg19.tar.gz 

file_name = './vgg19/model.onnx'
try:
    onnx_model = onnx.load(file_name)
except Exception as e:
    raise FrameworkError(
        'Cannot read the model file: "{}" is incorrect ONNX model file. Details: {}',
        file_name,
        str(e)
    ) from e

# maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
data_nodes_map = {}

for node in onnx_model.graph.input:
    print(str(node))
    name = str(node.name)
    data_nodes_map[name] = (name, 0)
    # print("name="+name)
    print('***************')

INTIALIZERS=onnx_model.graph.initializer
for initializer in INTIALIZERS:
    W= numpy_helper.to_array(initializer)
    print(initializer.name)
    print(W.shape)
    if W.ndim == 4:
        if initializer.name == 'conv1_1_w_0':
            W = np.swapaxes(W, 2, 3)  #swap w,h
            W = W[:, ::-1, :, :]     #revert RGB channel
        else:
            W = np.swapaxes(W, 2, 3)
        v_1d = W.reshape(W.shape[0] * W.shape[1] * W.shape[2] * W.shape[3])
        # for value in v_1d:
        #     print(value, end=" --- ")
        #     print(struct.pack('<f', value).hex())
        # if initializer.name == 'conv1_2_w_0':
        #     for value in v_1d:
        #         print(value, end=" --- ")
        #         print(struct.pack('<f', value).hex())
        print('---------split---------')
        # np.save(initializer.name, v_1d)
        v_1d.tofile('cnn_CnnMain_'+initializer.name)
    elif W.ndim == 1:
        W.tofile('cnn_CnnMain_'+initializer.name)
        pass
    else:  #W.ndim == 2
        print('convert error:'+str(W.ndim))
        if initializer.name == 'fc6_w_0':
            W = W.reshape(( W.shape[0] * W.shape[1])//49,7,7)  #swap w,h
            W = np.swapaxes(W, 1, 2)

            v_1d = W.reshape(W.shape[0] * W.shape[1] * W.shape[2])
        else:
            v_1d = W.reshape(W.shape[0] * W.shape[1])
        W.tofile('cnn_CnnMain_'+initializer.name)
        pass
    # print(W[0,0,:,:])
    #data_nodes_map[initializer] = (initializer, 0)
    print('xxxxxxxxxxxxxxx')

for node in onnx_model.graph.node:
    print(str(node))
    print('====================')
    # add incoming edges based on data_nodes_map
    for dst_port, inp in enumerate(node.input):
        # should add edge inp --> id
        if inp not in data_nodes_map:
            if inp == '':
                # input is omitted; most likely it corresponds to an optional input for an operator
                continue
            else:
                print(
                    'Reference to {} is not satisfied. A node refer not existing data tensor. ONNX model is not '
                    'consistent. Protobuf fragment: ', inp, '===', node)
                pass

        src_id, src_port = data_nodes_map[inp]

        edge_attrs = {
            'out': src_port,
            'in': dst_port,
            'name': inp,
            'fw_tensor_debug_info': [(inp, inp)],
            'in_attrs': ['in', 'name'],
            'out_attrs': ['out', 'name'],
            'data_attrs': ['fw_tensor_debug_info']
        }
        # print(str(edge_attrs))
        print('+++++++++++++++++++')
    print('________________')
    # add outgoing edges to data_nodes_map
    for src_port, out in enumerate(node.output):
        if out in data_nodes_map:
            print("Detected reuse of blob {}.".format(out))
        data_nodes_map[out] = (id, src_port)

		
