from scipy.io import loadmat 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import utility

def load_dataset(datasetName):
    # return a dictionary with X as key
    return loadmat(datasetName)

def display_photo(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()

def initialize_parameters():
    pass