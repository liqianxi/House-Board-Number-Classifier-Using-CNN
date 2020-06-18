from scipy.io import loadmat 
from matplotlib import pyplot as plt 
import numpy as np 
import utility
import tensorflow as tf
def model():
    dataset = utility.load_dataset('dataset/train_32x32.mat')
    testset = utility.load_dataset('dataset/test_32x32.mat')
    # use dataset['X'][:,:,:,i] to get ith picture
    training_data = dataset['X'] # (32, 32, 3, 73257)
    training_label = dataset['y'] # (73257, 1)
    testset_data = testset['X']  # (32, 32, 3, 26032)
    testset_label = testset['y']  # (26032, 1)

    # utility.display_photo(training_data)
    # training_data.shape 
    #training_x_flatten = training_data.reshape(training_data.shape[-1],-1).T
    #testset_x_flatten = testset_data.reshape(testset_data.shape[-1],-1).T
    # standardize the rgb colors
    training_x = training_data/256
    test_x = testset_data/256

    '''
    1. Initialize parameters / Define hyperparameters
    2. Loop for num_iterations:
        a. Forward propagation
        b. Compute cost function
        c. Backward propagation
        d. Update parameters (using parameters, and grads from backprop) 
    4. Use trained parameters to predict labels
    '''




model()