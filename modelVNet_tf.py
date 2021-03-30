import tensorflow as tf
from tensorflow import keras

def encoding_block(X, filter_size, filters_num, block_type, stage, s = 1, X_skip=0):
    conv_name_base = 'conv_' + block_type + str(stage) + '_'
    bn_name_base = 'bn_' + block_type + str(stage)  + '_'
    
    #Weight generation
    initializer = tf.keras.initializers.glorot_uniform()
    
    kernel = tf.Variable(initializer(shape=(filter_size[0],filter_size[1],1,filters_num)))
    X = tf.nn.conv2d(X, kernel, strides = [1, s, s, 1], padding='SAME', name = conv_name_base + 'main_' + str(1))
    X = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + 'main_' + str(1))(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    # Down sampling layer
    kernel = tf.Variable(initializer(shape=(2,2,1,filters_num*2)))
    X_downed = tf.nn.conv2d(X, kernel, strides = [1, 2, 2, 1], padding='VALID', name = conv_name_base + 'down')
    X_downed = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + 'down')(X_downed)
    X_downed = tf.keras.layers. Activation('relu')(X_downed)
    return X, X_downed

def calculate_outshape(tensor):
    output_shape = tensor.shape.numpy()
    output_shape = output_shape[1] *2
    output_shape = output_shape[2] *2
    output_shape = output_shape[3] *0.5
    return output_shape

def decoding_block(X, filter_size, filters_num, block_type, stage, s = 1, X_jump = 0, up_sampling = True):
    conv_name_base = 'conv_' + block_type + str(stage) + '_'
    bn_name_base = 'bn_' + block_type + str(stage)  + '_'
    
    #Weight generation
    initializer = tf.keras.initializers.glorot_uniform()
    
    # Joining X_jump from encoding side with X_uped
    if tf.rank(X_jump) == 0:
        X_joined_input = X
    else:
        X_joined_input = tf.keras.layers.concatenate([X,X_jump],axis = 3)
    
    # First component of main path 
    kernel = tf.Variable(initializer(shape=(filter_size[0],filter_size[1],1,filters_num)))
    X_joined_input = tf.nn.conv2d(X_joined_input, kernel, strides = [1, s, s, 1], padding='SAME', name = conv_name_base + 'main_' + str(1))
    X_joined_input = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + 'main_' + str(1))(X_joined_input)
    X_joined_input = tf.keras.layers.Activation('relu')(X_joined_input)
    
    # Up-sampling layer. At the output layer, up-sampling is disabled and replaced by other stuffs manually
    if up_sampling == True:
        kernel = tf.Variable(initializer(shape=(2,2,1,filters_num)))
        output_shape = calculate_outshape(X)    
        X_uped = tf.nn.conv2d_transpose(X_joined_input, kernel, output_shape=output_shape, strides = [1, 2, 2, 1], padding = "VALID", name = conv_name_base + 'up')
        X_uped = tf.keras.layers.BatchNormalization(axis = 3, name = bn_name_base + 'up')(X_uped)
        X_uped = tf.keras.layers.Activation('relu')(X_uped)
        return X_uped
    else:
        return X_joined_input

def VNet(pretrained_weights = None, input_shape = (480,720,1), filter_size = (10,10)):
    
    # Input is 720x480 greyscale image(1 channel)
    X_input = keras.Input(input_shape)
    
    # Encoding Stream
    X_jump1, X_out = encoding_block(X = X_input, X_skip = 0, filter_size= filter_size, filters_num= 16,
                                    block_type = "down", stage = 1, s = 1)
    X_jump2, X_out = encoding_block(X = X_out, X_skip = X_out, filter_size= filter_size, filters_num= 32,
                                    block_type = "down", stage = 2, s = 1)
    X_jump3, X_out = encoding_block(X = X_out, X_skip = X_out, filter_size= filter_size, filters_num= 64,
                                    block_type = "down", stage = 3, s = 1)
    X_jump4, X_out = encoding_block(X = X_out, X_skip = X_out, filter_size= filter_size, filters_num= 128,
                                    block_type = "down", stage = 4, s = 1)
    # Decoding Stream
    X_out = decoding_block(X = X_out, X_jump = 0, filter_size= filter_size, filters_num= 256, 
                           block_type = "up", stage = 1, s = 1)
    X_out = decoding_block(X = X_out, X_jump = X_jump4, filter_size= filter_size, filters_num= 256, 
                           block_type = "up", stage = 2, s = 1)
    X_out = decoding_block(X = X_out, X_jump = X_jump3, filter_size= filter_size, filters_num= 128, 
                           block_type = "up", stage = 3, s = 1)
    X_out = decoding_block(X = X_out, X_jump = X_jump2, filter_size= filter_size, filters_num= 64, 
                           block_type = "up", stage = 4, s = 1)
    X_out = decoding_block(X = X_out, X_jump = X_jump1, filter_size= filter_size, filters_num= 32, 
                           block_type = "up", stage = 5, s = 1, up_sampling = False)
    # Output layer operations
    X_out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1,1) , strides = (1,1), padding = 'valid',
                   name = "conv_out", kernel_initializer = tf.keras.initializers.glorot_uniform())(X_out)
    X_out = tf.keras.layers.Activation('softmax')(X_out)
    model = keras.Model(inputs=X_input, outputs=X_out, name="VNet")
    return model
    
model = VNet()

model.summary()
tf.keras.utils.plot_model(model, "VNet.png", show_shapes=True)