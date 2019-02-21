import pickle
import sys
import os
import glob
#import cv2
import math
import pickle
import datetime
import pandas as pd
from pandas import DataFrame
from pandas import concat
from PIL import Image
import numpy as np
import scipy.misc
from PIL import Image
from sklearn.decomposition import PCA

import bcolz
#from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.utils import plot_model

#n_inputs = 30
#n_outputs = 7
processed_folder =  "processeddata/"
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def saveFile(fileName, object):
	with open(fileName,'wb') as f:
 	   pickle.dump(object,f)

def loadFile(fileName):
	fileObject = open(fileName,'rb')
	return pickle.load(fileObject,encoding='bytes')


def unscale(y_arr, scaler, template_df, cols_to_scale, y_cols, toint=False):
    """
    Unscale array y_arr of model predictions, based on a scaler fitted
    to template_df.
    """
    tmp = template_df.copy()
    tmp[y_cols] = pd.DataFrame(y_arr, index=tmp.index)
    tmp[cols_to_scale] = scaler.inverse_transform(tmp[cols_to_scale])
    if toint:
        return tmp[y_cols].astype(int)
    return tmp[y_cols]

def vector_smape(y_pred, y_real):
    nom = np.abs(y_pred-y_real)
    denom = (np.abs(y_pred) + np.abs(y_real)) / 2
    results = nom / denom
    return 100*np.mean(results)
