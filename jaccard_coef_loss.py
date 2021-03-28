import tensorflow
import numpy as np
import matplotlib.pyplot as plt


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true * y_pred), axis=-1)
    sum_ = tensorflow.math.reduce_sum(tensorflow.math.abs(y_true)
                                      + tensorflow.math.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return tensorflow.stack((1 - jac) * smooth)


# Test and plot
y_pred = np.array([np.arange(-10, 10+0.1, 0.1)]).T
y_true = np.zeros(y_pred.shape)
name='jaccard_distance_loss'
try:
    loss = jaccard_distance_loss(
        tensorflow.Variable(y_true),
        tensorflow.Variable(y_pred))
except Exception as e:
    print("error plotting", name ,e)
else:
    plt.title(name)
    plt.plot(y_pred,loss)
    plt.show()
    
# Test
# Test
print("TYPE                 |Almost_right |half right |all_wrong")
y_true = np.array([[0,0,1,0],[0,0,1,0],[0,0,1.,0.]])
y_pred = np.array([[0,0,0.9,0],[0,0,0.1,0],[1,1,0.1,1.]])

r = jaccard_distance_loss(
    tensorflow.Variable(y_true),
    tensorflow.Variable(y_pred),)
print('jaccard_distance_loss',r)
assert r[0]<r[1]
assert r[1]<r[2]

r = tensorflow.keras.losses.binary_crossentropy(tensorflow.Variable(y_true),
                                                tensorflow.Variable(y_pred))
print('binary_crossentropy',r)
print('binary_crossentropy_scaled',r/tensorflow.math.reduce_max(r))
assert r[0]<r[1]
assert r[1]<r[2]

"""
Keraas backend old
TYPE                 |Almost_right |half right |all_wrong
jaccard_distance_loss [ 0.09900928  0.89108944  3.75000238]
binary_crossentropy [  0.02634021   0.57564634  12.53243446]
binary_crossentropy_scaled [ 0.00210176  0.04593252  1.        ]
"""

"""
tensorflow
TYPE                 |Almost_right |half right |all_wrong
jaccard_distance_loss tf.Tensor([0.0990099  0.89108911 3.75      ], shape=(3,), dtype=float64)
binary_crossentropy tf.Tensor([ 0.0263401   0.57564602 12.14435738], shape=(3,), dtype=float64)
binary_crossentropy_scaled tf.Tensor([0.00216892 0.04740029 1.        ], shape=(3,), dtype=float64)
"""