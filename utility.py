from scipy.io import loadmat 
from matplotlib import pyplot as plt 
import numpy as np 


def load_dataset(datasetName):
    # return a dictionary with X as key
    return loadmat(datasetName)
    #print(len(annots['X']))
    #print(len(annots['X'][0]))
    #width, height = 32, 32
    #channels = len(annots['X'][0][0])
    #training_num = len(annots['X'][0][0][0])
    #print(training_num)
    #plt.imshow(annots['X'][:,:,:,3200], interpolation='nearest')# display index 200 picture
    #plt.show()

def display_photo(picture):
    plt.imshow(picture, interpolation='nearest')
    plt.show()
