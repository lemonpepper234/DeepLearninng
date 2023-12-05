import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from tools import *

with h5py.File("C:\\sciencificcomputation\\code\\deeplearning\\week4\\myModel\\datasets\\train_catvnoncat.h5", "r") as hdf:
    train_x_orig = np.array(hdf["train_set_x"][:])
    train_y_orig = np.array(hdf["train_set_y"][:])
    train_y_orig = train_y_orig.reshape((1, train_x_orig.shape[0]))

with h5py.File('C:\\sciencificcomputation\\code\\deeplearning\\week4\\myModel\\datasets\\test_catvnoncat.h5', "r") as hdf:
    test_x_orig = np.array(hdf["test_set_x"][:])
    test_y_orig = np.array(hdf["test_set_y"][:])
    test_y_orig = test_y_orig.reshape((1, test_y_orig.shape[0]))

train_x  = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_y = train_y_orig
test_y = test_y_orig

#* standralize
train_x = train_x / 255.
test_x = test_x / 255.

#* the initial parameters
learning_rate = 0.0075
sizes_of_layers = [train_x.shape[0], 50, 30, 20, 7, 5, 1]
iteration_num = 6000
activation = "relu"

final_parameters = L_layer_model_training(train_x, train_y, sizes_of_layers, activation, learning_rate, iteration_num, print_cost = True)