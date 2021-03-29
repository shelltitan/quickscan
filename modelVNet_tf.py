import tensorflow as tf

initializer = tf.keras.initializers.glorot_uniform()
kernel = tf.Variable(initializer(shape=(10,10,1,16)))
#X = tf.nn.conv2d(X, kernel, strides = [1, s, s, 1], padding='SAME', name = conv_name_base + 'main_' + str(1))