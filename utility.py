from scipy.io import loadmat 
from matplotlib import pyplot as plt 
import numpy as np 


def load_dataset(trainName, testName):
    # return a dictionary with X as key
    full_train_set = loadmat(trainName)
    full_test_set = loadmat(testName)
    trainX, trainY = full_train_set['X'], full_train_set['y']
    testX, testY = full_test_set['X'], full_test_set['y']
    return [trainX, trainY, testX, testY]

def display_photo(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()
