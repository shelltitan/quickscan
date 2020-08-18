import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def VNet(pretrained_weights = None, input_shape = (240,320,1), filter_size = (10,10)):
    
    # Input is 320x240 greyscale image(1 channel)
    inputs = keras.Input(input_shape)
    """Down Sample Side"""
    # 1st layer convulution and batch normalization with Relu
    layer1 = tf.keras.layers.Conv2D(filters = 16, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(inputs)
    layer1 = tf.keras.layers.BatchNormalization(axis = 3)(layer1)
    layer1 = tf.keras.layers.Activation('relu')(layer1)
    
    #1st layer downsampling and batch normalization with Relu
    layer1d = tf.keras.layers.Conv2D(filters = 32, kernel_size = (2,2), strides = (2,2),
                                    padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer1)
    layer1d = tf.keras.layers.BatchNormalization(axis = 3)(layer1d)
    layer1d = tf.keras.layers.Activation('relu')(layer1d)
    
    #2nd layer convulution and batch normalization with Relu
    layer2 = tf.keras.layers.Conv2D(filters = 32, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer1d)
    layer2 = tf.keras.layers.BatchNormalization(axis = 3)(layer2)
    layer2 = tf.keras.layers.Activation('relu')(layer2)
    
    #2nd layer downsampling and batch normalization with Relu
    layer2d = tf.keras.layers.Conv2D(filters = 64, kernel_size = (2,2), strides = (2,2),
                                    padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer2)
    layer2d = tf.keras.layers.BatchNormalization(axis = 3)(layer2d)
    layer2d = tf.keras.layers.Activation('relu')(layer2d)
    
    #3rd layer convulution and batch normalization with Relu
    layer3 = tf.keras.layers.Conv2D(filters = 64, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer2d)
    layer3 = tf.keras.layers.BatchNormalization(axis = 3)(layer3)
    layer3 = tf.keras.layers.Activation('relu')(layer3)
    
    #3rd layer downsampling and batch normalization with Relu
    layer3d = tf.keras.layers.Conv2D(filters = 128, kernel_size = (2,2), strides = (2,2),
                                    padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer3)
    layer3d = tf.keras.layers.BatchNormalization(axis = 3)(layer3d)
    layer3d = tf.keras.layers.Activation('relu')(layer3d)
    
    #4th layer convulution and batch normalization with Relu
    layer4 = tf.keras.layers.Conv2D(filters = 128, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer3d)
    layer4 = tf.keras.layers.BatchNormalization(axis = 3)(layer4)
    layer4 = tf.keras.layers.Activation('relu')(layer4)
    
    #4th layer downsampling and batch normalization with Relu
    layer4d = tf.keras.layers.Conv2D(filters = 256, kernel_size = (2,2), strides = (2,2),
                                    padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer4)
    layer4d = tf.keras.layers.BatchNormalization(axis = 3)(layer4d)
    layer4d = tf.keras.layers.Activation('relu')(layer4d)
    
    #5th layer convulution and batch normalization with Relu
    layer5 = tf.keras.layers.Conv2D(filters = 256, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer4d)
    layer5 = tf.keras.layers.BatchNormalization(axis = 3)(layer5)
    layer5 = tf.keras.layers.Activation('relu')(layer5)
    
    """Up Sample Side"""
    # 5th layer upsampling and batch normalization with Relu
    layer5u = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2),
                                              padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer5)
    layer5u = tf.keras.layers.BatchNormalization(axis = 3)(layer5u)
    layer5u = tf.keras.layers.Activation('relu')(layer5u)
    
    #Concating layer 5 and layer 4
    layer5c4 = tf.keras.layers.concatenate([layer5u,layer4],axis = 3)
    
    # 4th layer convulution
    layer4 = tf.keras.layers.Conv2D(filters = 256, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer5c4)
    layer4 = tf.keras.layers.BatchNormalization(axis = 3)(layer4)
    layer4 = tf.keras.layers.Activation('relu')(layer4)
    
    # 4th layer upsampling and batch normalization with Relu
    layer4u = tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2),
                                              padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer4)
    layer4u = tf.keras.layers.BatchNormalization(axis = 3)(layer4u)
    layer4u = tf.keras.layers.Activation('relu')(layer4u)
    
    #Concating layer 4 and layer 3
    layer4c3 = tf.keras.layers.concatenate([layer4u,layer3], axis = 3)
    
    # 3rd layer convulution
    layer3 = tf.keras.layers.Conv2D(filters = 128, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer4c3)
    layer3 = tf.keras.layers.BatchNormalization(axis = 3)(layer3)
    layer3 = tf.keras.layers.Activation('relu')(layer3)
    
    # 3rd layer upsampling and batch normalization with Relu
    layer3u = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (2,2), strides = (2,2),
                                              padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer3)
    layer3u = tf.keras.layers.BatchNormalization(axis = 3)(layer3u)
    layer3u = tf.keras.layers.Activation('relu')(layer3u)
    
    #Concating layer 3 and layer 2
    layer3c2 = tf.keras.layers.concatenate([layer3u,layer2], axis = 3)
    
    # 2nd layer convulution
    layer2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer3c2)
    layer2 = tf.keras.layers.BatchNormalization(axis = 3)(layer2)
    layer2 = tf.keras.layers.Activation('relu')(layer2)
    
    # 2nd layer upsampling and batch normalization with Relu
    layer2u = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (2,2), strides = (2,2),
                                              padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer2)
    layer2u = tf.keras.layers.BatchNormalization(axis = 3)(layer2u)
    layer2u = tf.keras.layers.Activation('relu')(layer2u)
    
    #Concating layer 2 and layer 1
    layer2c1 = tf.keras.layers.concatenate([layer2u,layer1], axis = 3)
    
    # 1st layer convulution
    layer1 = tf.keras.layers.Conv2D(filters = 32, kernel_size = filter_size, strides = (1,1),
                                    padding = 'same', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer2c1)
    layer1 = tf.keras.layers.BatchNormalization(axis = 3)(layer1)
    layer1 = tf.keras.layers.Activation('relu')(layer1)
    
    # Output layer operations
    out = tf.keras.layers.Conv2D(filters = 3,kernel_size = (1,1), strides = (1,1),
                                    padding = 'valid', kernel_initializer = tf.keras.initializers.GlorotUniform())(layer1)
    out = tf.keras.layers.Activation('softmax')(out)
    model = keras.Model(inputs=inputs, outputs=out, name="VNet")
    
    return model
    
model = VNet()

model.summary()
tf.keras.utils.plot_model(model, "VNet.png", show_shapes=True)


