#Link to model / weights - https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

import sys
import numpy as np
sys.path.append('C:\\temp_20151027\\model_optimizer')

from mo.front.caffe.proto import caffe_pb2


def get_layers(proto):
    if len(proto.layer):
        return proto.layer
    elif len(proto.layers):
        return proto.layers
    else:
        print('Invalid proto file: there is neither "layer" nor "layers" top-level messages. ')

#the model is downloaded from http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel 
caffemodel_filename = './vgg16/VGG_ILSVRC_16_layers.caffemodel'
model = caffe_pb2.NetParameter()
f = open(caffemodel_filename, 'rb')
model.ParseFromString(f.read())
f.close()

# print(str(model))
proto_layers = get_layers(model)
for i, layer in enumerate(proto_layers):
    print('-----------  i='+str(i))
    # print(str(layer))
    print(layer.name)
    print(layer.type)
    for j, node in enumerate(layer.blobs):
        print('     -----------  j=' + str(j))
        print(node.num)
        print(node.channels)
        print(node.height)
        print(node.width)
        # print(type(node))
        # print(type(node.data))
        # print(type(node.data[0]))
        ndim = 0
        if node.num != 1:
            ndim += 1
        if node.channels != 1:
            ndim += 1
        if node.height != 1:
            ndim += 1
        if node.width != 1:
            ndim += 1
        # print(len(node.data))

        W = np.asarray(node.data, dtype=np.float32)

        if ndim == 4:
            W = np.reshape(W, (node.num, node.channels, node.width, node.height))
            if j == 0:  # Weights
                if layer.name == 'conv1_1':
                    W = np.swapaxes(W, 2, 3)  # swap w,h
                    W = W[:, ::-1, :, :]    #revert RGB channel
                    print('transform conv1_1')
                else:
                    W = np.swapaxes(W, 2, 3)
        elif ndim == 1:
            pass
        else:  # ndim == 2
            W = np.reshape(W, (node.width, node.height))
            if j == 0:  # Weights
                if layer.name == 'fc6':
                    print('transform fc6')
                    # print(W.shape[0])
                    # print(W.shape[1])
                    W = np.reshape(W, ((W.shape[0] * W.shape[1]) // 49, 7, 7))  # swap w,h
                    # print(W.shape)
                    W = np.swapaxes(W, 1, 2)

                    W = W.reshape(W.shape[0] * W.shape[1] * W.shape[2])
                else:
                    W = W.reshape(W.shape[0] * W.shape[1])
            pass

        if j==0 :
            W.tofile(layer.name + '_w')
        elif j == 1:
            W.tofile(layer.name + '_b')
        else:
            W.tofile(layer.name + '_' + str(j))
        # print(W.shape)


    # print(type(node.data))

print('ok')

