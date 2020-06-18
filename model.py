from scipy.io import loadmat 
from matplotlib import pyplot as plt 
import numpy as np 
import utility
import tensorflow as tf
def model():
    dataset = utility.load_dataset('dataset/train_32x32.mat')
    # use dataset['X'][:,:,:,i] to get ith picture
    training_data = dataset['X']
    training_label = dataset['y']
    # utility.display_photo(training_data)
    # training_data.shape (32, 32, 3, 73257)
    # training_x_flatten: (32*32*3, 73257)
    training_x_flatten = training_data.reshape(training_data.shape[-1],-1).T
    
    

model()