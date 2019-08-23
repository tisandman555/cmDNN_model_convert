# cmDNN_model_convert
This project is used for converting the models from public model zoos to cmDNN required data format

Currently, it can convert 4 models: VGG16/VGG19/Alexnet/Resnet50. The project is based on OpenVINO 2019R1 project. It is using the part of model optimizer to parse the model file, then do the layer fusion if it is necessary and output the weights and bias seperately.
