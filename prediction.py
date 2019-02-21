#predict
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import os, sys, time, math, json, random
#import tensorflow as tf
import matplotlib.pyplot as plt
#from keras.utils import np_utils
#%matplotlib inline
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Concatenate, concatenate, Input, Reshape, LSTM, TimeDistributed
from keras.layers.embeddings import Embedding
#import seaborn as sns
import warnings
from keras import models, layers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


def stateful_prediction( mm,X_test, ntarget=1):
        #expecting..
        bis = mm.layers[0].get_config()["batch_input_shape"]
        batch_size, ts, nfeat = bis
        assert(X_test.shape[0] % batch_size == 0)
        assert(X_test.shape[1] % ts == 0)

        y_pred = np.zeros((X_test.shape[0],X_test.shape[1],ntarget))
        #y_pred[:] = np.NaN
        for ipat in range(0, X_test.shape[0],batch_size):
            mm.reset_states()
            x_next = X_test[ipat:(ipat+batch_size),0,:]
            x_next = x_next.reshape( x_next.shape[0], 1, x_next.shape[1])
            #print("x_next", x_next.shape)
            for itime in range(0, X_test.shape[1], ts):
                y_pred[ipat:(ipat+batch_size),itime:(itime+ts),:] = mm.predict(x_next,batch_size = batch_size)
                y_pred_temp =  np.squeeze(y_pred[ipat:(ipat+batch_size),itime:(itime+ts),:])
                y_pred_temp= y_pred_temp.reshape((batch_size ,1))
                #print( X_testi[:,:,0].shape, y_pred_temp.shape)
                #sys.exit()
                try:
                    x_next = X_test[ipat:(ipat+batch_size),(itime+ts):(itime+2*ts),:]
                    #print("x_next", x_next.shape)
                    x_next[:,:,0] = y_pred_temp#y_pred[ipat:(ipat+batch_size),itime:(itime+ts),:][:,:]
                    #print("x_next3",x_next.shape )
                except IndexError:
                    pass
        return y_pred

def stateless_prediction(mm, X_test, batch_size):
    n_time = X_test.shape[1]
    y_pred = np.zeros((X_test.shape[0],X_test.shape[1],1))
    x_next = X_test[:,0]

    for i in range(0, n_time):
        #mm.reset_states()
        x_arr = x_next.reshape( x_next.shape[0], 1, x_next.shape[1] )
        #print(x_arr.shape)
        #print(y_pred[:,i,:].shape)
        y_pred_temp = mm.predict(x_arr,batch_size = batch_size) # input for prediction must be 2d, output is immediately extracted from 2d to 1d
        #print(x_arr[:5])
        #print(y_pred_temp.shape, y_pred_temp)
        y_pred[:,i,:] = np.squeeze(y_pred_temp).reshape((500,1))
        #print(y_pred[:,i,:] ,y_pred[:,i,:].shape)
        #sys.exit()
        try:
            x_next = X_test[:,i+1]
            x_next[:,0] = np.squeeze(y_pred_temp)
            #print(x_next.shape )
            #sys.exit()
        except IndexError:
            pass  # this happens on last iteration, and x_next does not matter anymore

    return y_pred
