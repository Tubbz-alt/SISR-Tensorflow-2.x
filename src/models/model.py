import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU


def convolution_block(inputs, n_filters, filter_size, strides):
    x = Conv2D(n_filters, kernel_size=filter_size, strides= strides)(inputs)
    x = BatchNormalization()(x)
    return LeakyReLU(x)





#testing for errors
print(convolution_block(tf.ones((1,2,2,3)), 64,3,1))

