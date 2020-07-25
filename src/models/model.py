import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, Input, Lambda
from utils import norm11



def convolution_block(inputs, n_filters, filter_size, strides, bn = True, momentum = 0.8):
    x = Conv2D(n_filters, kernel_size=filter_size, strides= strides, padding = "same")(inputs)
    if bn:
        x = BatchNormalization(momentum= momentum)(x)
    return LeakyReLU(alpha=0.2)(x)

def discriminator():

    x_input = Input(shape = (96,96,3))
    x = Lambda(norm11)(x_input)

    x = convolution_block(x, 64, 3, 1, False)
    x = convolution_block(x, 64, 3, 1)
    x = convolution_block(x, 128, 3, 1)
    x = convolution_block(x, 128, 3, 2)
    x = convolution_block(x, 256, 3, 1)
    x = convolution_block(x, 256, 3, 2)
    x = convolution_block(x, 512, 3, 1)
    x = convolution_block(x, 512, 3, 2)

    x = Flatten()(x)


    x - Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation = "sigmoid")

    return x







#testing for errors
print(convolution_block(tf.ones((1,2,2,3)), 64,3,1))

print("1 .TESTED \n \n")

print(discriminator())
