import tensorflow as tf 
import numpy as np 


# Normalizations 


def norm11(x):
    return x/127.5 - 1
