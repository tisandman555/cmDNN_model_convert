import sys
sys.path.append('C:\\temp_20151027\\model_optimizer')

import tensorflow as tf
from mo.front.tf.loader import load_tf_graph_def, protobuf2nx

import numpy as np
import struct

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])
def double_to_hex(f):
    return hex(struct.unpack('<Q', struct.pack('<d', f))[0])

graph_def, variables_values = load_tf_graph_def(graph_file_name="./tf_model/Resnet50.pb", is_binary=True,
                                                    checkpoint="",
                                                    user_output_node_names_list=[],
                                                    model_dir="",
                                                    meta_graph_file="",
                                                    saved_model_tags=[])
#import the GraphDef to the global default Graph
tf.import_graph_def(graph_def, name='')
# extract all the constant ops from the Graph
# and run all the constant ops to get the values (weights) of the constant ops
constant_values = {}
with tf.Session() as sess:
    for node in tf.get_default_graph().as_graph_def().node:
        # print(node)
        if 'epsilon' in node.attr.keys():
            print(node.name, node.attr['epsilon'].f)
        # print('--------------------------------------------------')

    # for op in sess.graph.get_operations():
    #     print(str(op))
    #     print('--------------------------------------------------')

    # constant_ops = [op for op in sess.graph.get_operations() if op.type == "FusedBatchNorm"]
    # for constant_op in constant_ops:
    #     if 'strides' in constant_ops.attr.keys():
    #         print(constant_ops.name, [a for a in constant_ops.attr['strides'].list.i])

    constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
    for constant_op in constant_ops:
        value =  sess.run(constant_op.outputs[0])
        # constant_values[constant_op.name] = value
        #In most cases, the type of the value is a numpy.ndarray.
        #So, if you just print it, sometimes many of the values of the array will
        #be replaced by ...
        #But at least you get an array to python object, 
        #you can do what other you want to save it to the format you want

        print(constant_op.name)

        print("dim= "+str(value.ndim))
        vshape = value.shape[:]
        for s in vshape:
            print("     dim: "+str(s))
        # print(value)

        # '''
        s = constant_op.name
        s1 = s.split('/')
        print(s1[0])   #"conv1"
        print(s1[1])    #"kernel"
 
        if s1[1] == 'kernel':
            fn_prefix = s1[0]
            W = value
        elif s1[1] == 'bias':
            b = value
        elif s1[1] == 'gamma':
            gamma = value
        elif s1[1] == 'beta':
            beta = value
        elif s1[1] == 'moving_mean':
            mean = value
        elif s1[1] == 'moving_variance':
            var = value
            
            print('Weight = ')
            print(W.shape)
            # print(W)
            print('bias = ')
            print(b.shape)
            # print(b)
            print('gamma = ')
            # print(gamma)
            print(gamma.shape)
            print('beta = ')
            print(beta.shape)
            # print(beta)
            print('mean = ')
            print(mean.shape)
            # print(mean)
            print('var = ')
            print(var.shape)
            # print(var)
            #eps = 0.0010000000474974513
            eps = 0.001
            # print(type(eps))
            print(str(eps))

            b_bn = beta - gamma*mean/np.sqrt(var+eps)
        
            w_bn = np.diag(gamma/np.sqrt(var+eps))
            print('w_bn.shape='+str(w_bn.shape))

            W = np.swapaxes(W, 1, 2)
            W = np.swapaxes(W, 0, 3)
            W_shape = W.shape
            W = np.reshape(W, (b.shape[0], -1))
            print('W.shape=' + str(W.shape))
            print(W)
            w_new_2 = w_bn
        
            w_new_1 = np.matmul( w_bn, W )
            b_new_1 = b + b_bn

            print('New bias 1 = ')
            print(b_new_1.shape)
            # print(b_new_1)
            b_new_1.tofile('cnn_CnnMain_'+fn_prefix+'_b')
        
            w_new_1 = np.reshape(w_new_1, W_shape)
            print('w_new_1.shape=' + str(w_new_1.shape))
            # print(w_new_1)

            if(fn_prefix == 'conv1'):
                print('revert RGB channel')
                w_new_1 = w_new_1[:, ::-1, :, :]
            print('New Weight 1= ')
            print(w_new_1.shape)
            # print(w_new_1)
            w_new_1.tofile('cnn_CnnMain_'+fn_prefix+'_w')

        print("*****************")

        if constant_op.name == 'fc1000/kernel':
            s = constant_op.name
            s1 = s.split('/')
            print(type(s1))
            print(s1[0])
            print(s1[1])
            s2 = s.find('/')
            print(s2)
            s2 = s1[0].find('/')
            print(s2)
            W = value
            print(W.shape)
            W = np.swapaxes(W, 0, 1)
        elif constant_op.name == 'fc1000/bias':
            b = value

            print('Weight = ')
            print(W.shape)
            # print(W)
            print('bias = ')
            print(b.shape)
            # print(b)


            b.tofile('cnn_CnnMain_fc1000_b')

            W.tofile('cnn_CnnMain_fc1000_w')

            break
        print("*****************")
        