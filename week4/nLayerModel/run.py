import numpy as np
import h5py
import matplotlib.pyplot as plt
from tools import *
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print("sb")