from scipy.io import loadmat 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import utility


class Model:
    def __init__(self):
        self.__model = 
    def model_definition(self):
        # empty model
        model = Sequential()

        # first Conv layer
        model.add(Conv2D(filters=15, input_shape=(32,32,3), kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # second Conv layer
        model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(4,4), strides=(2,2), padding='valid'))

        # third conv layer
        model.add(Conv2D(filters=60, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))

        # FC part
        model.add(Flatten())
        # first FC layer
        model.add(Dense(960))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        # second FC layer
        model.add(Dense(240))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        # third FC layer
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))

        # Output
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.summary()






    def learning(self):
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
        model = 

        '''
        1. Initialize parameters / Define hyperparameters
        2. Loop for num_iterations:
            a. Forward propagation
            b. Compute cost function
            c. Backward propagation
            d. Update parameters (using parameters, and grads from backprop) 
        4. Use trained parameters to predict labels
        '''




    