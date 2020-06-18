from scipy.io import loadmat 
from matplotlib import pyplot as plt 
import numpy as np 
import tensorflow as tf

def load_dataset(datasetName):
    # return a dictionary with X as key
    return loadmat(datasetName)

def display_photo(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()

def create_placeholder(height, width, channels, y_Num):
    X = tf.placeholder(dtype="float", shape = [None, height, width,channels])
    Y = tf.placeholder(dtype="float",shape=[None, y_Num])
    return X, Y