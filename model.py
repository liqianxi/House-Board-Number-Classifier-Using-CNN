from scipy.io import loadmat 
from matplotlib import pyplot as plt 
import numpy as np 
import utility
def model():
    dataset = utility.load_dataset('dataset/train_32x32.mat', 'dataset/test_32x32.mat')
    # use dataset['X'][:,:,:,i] to get ith picture
    test_picture = dataset[0][:,:,:,4]
    utility.display_photo(test_picture)

model()