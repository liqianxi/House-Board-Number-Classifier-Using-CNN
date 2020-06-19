from scipy.io import loadmat 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utility


class Model:
    def __init__(self):
        self.model = self.model_definition()
        self.training_x = None
        self.training_y = None
        self.test_x = None
        self.test_y = None
        
    def model_definition(self):
        # empty model
        
        model = Sequential()
        
        # first Conv layer
        model.add(Conv2D(filters=15, input_shape=(32,32,3), kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        
        # second Conv layer
        model.add(Conv2D(filters=25, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(4,4), strides=(2,2), padding='valid'))
        
        # third conv layer
        model.add(Conv2D(filters=60, kernel_size=(1,1), strides=(1,1), padding='same'))
        model.add(Activation('relu'))
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
        model.add(Dense(11))
        
        model.add(Activation('softmax'))

        #model.summary()
        return model

    def load_all_data(self):
        dataset = utility.load_dataset('dataset/train_32x32.mat')
        testset = utility.load_dataset('dataset/test_32x32.mat')
        # use dataset['X'][:,:,:,i] to get ith picture
        training_data = dataset['X'] 
        transform_x = np.rollaxis(training_data, axis=-1)  # (73257, 32, 32, 3)
        training_label = to_categorical(dataset['y']) # (73257, 1)
        testset_data = testset['X']  # (32, 32, 3, 26032)
        transform_test_x = np.rollaxis(testset_data, axis=-1)  # (26032, 32, 32, 3)
        testset_label = to_categorical(testset['y'])  # (26032, 1)
        # standardize the rgb colors
        self.training_x = transform_x/256
        self.test_x = transform_test_x/256
        self.training_y = training_label
        self.test_y = testset_label

    def learning(self):
        assert self.training_x != None,"X is None"
        assert self.training_y != None,"Y is None"
        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.training_x, self.training_y, batch_size=32, epochs=10, shuffle=True)
        self.model.save('/model')
        '''
        1. Initialize parameters / Define hyperparameters
        2. Loop for num_iterations:
            a. Forward propagation
            b. Compute cost function
            c. Backward propagation
            d. Update parameters (using parameters, and grads from backprop) 
        4. Use trained parameters to predict labels
        '''
    def predict(self):
        y_predict = self.model.predict_classes(self.test_x)
        accuracy = tf.keras.losses.BinaryCrossentropy()(y_true=self.test_y,y_pred=y_predict).numpy()
        print("accuracy rate for test set is:", accuracy)


model = Model()
model.load_all_data()
try:
    keras.models.load_model("/model")
except Exception:
    print("Model loading Failed, learn again")
    model.learning()
else:
    print("Model loading successful")
finally:
    model.predict()


'''
dataset = utility.load_dataset('dataset/train_32x32.mat')
print(dataset['X'][:,:,:,0])    
temp = np.rollaxis(dataset['X'], axis=-1)[0,:,:,:]
print(temp.shape) '''

