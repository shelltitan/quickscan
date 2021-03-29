import tensorflow as tf
import numpy as np

tf.random.set_seed(45)

x_in = np.random.randn(1,480,720,1) # [batch, in_height, in_width, in_channels]

test = tf.constant(x_in, dtype=tf.float32)

# strides = [batch, height, width, channels]
#shape=[10, 10, 1, 16], # [filter_height, filter_width, in_channels, out_channels]
initializer = tf.keras.initializers.glorot_uniform()

#tf.print(initializer(shape=(10,10,1,16)).shape)

kernel = tf.Variable(initializer(shape=(10,10,1,16)))
tf.print(test.shape)
test = tf.nn.conv2d(test, kernel, strides = [1, 1, 1, 1], padding='SAME', name = "test")
test = tf.nn.batch_normalization(test,name = "test2")
tf.print(test.shape)
