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


def stateful_prediction( mm, X_all,X_test, X6_testi, ntarget=1):
        #expecting..
        [X1_all,X2_all,X3_all,X4_all,X5_all,X6_all] = X_all
        [X1_test,X2_test,X3_test,X4_test,X5_test] = X_test
        bis = mm.layers[0].get_config()["batch_input_shape"]
        batch_size, ts, nfeat = bis
        assert(X1_all.shape[0] % batch_size == 0)
        assert(X1_all.shape[1] % ts == 0)

        y_pred = np.zeros((X1_test.shape[0],X1_test.shape[1],ntarget))
        y_pred[:] = np.NaN

        for ipat in range(0,X1_test.shape[0],batch_size):
            mm.reset_states()

            for itime in range(0,X1_all.shape[1] + X1_test.shape[1], ts):
                #print("itime",  itime)
                if itime < X1_all.shape[1]:
                    X1_alli = X1_all[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X2_alli = X2_all[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X3_alli = X3_all[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X4_alli = X5_all[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X5_alli = X5_all[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X6_alli = X6_all[ipat:(ipat+batch_size),itime:(itime+ts),:]

                    X_alli = [X1_alli,X2_alli,X3_alli,X4_alli,X5_alli,X6_alli]

                    mm.predict(X_alli,batch_size=batch_size, steps = None)
                else:
                    itime = itime - X1_all.shape[1]
                    X1_testi = X1_test[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X2_testi = X2_test[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X3_testi = X3_test[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X4_testi = X5_test[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X5_testi = X5_test[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    #X6_testi = X6_test[ipat:(ipat+batch_size),itime:(itime+ts),:]
                    X_testi = [X1_testi,X2_testi,X3_testi,X4_testi,X5_testi,X6_testi]

                    y_pred[ipat:(ipat+batch_size),itime:(itime+ts),:] = mm.predict(X_testi,batch_size = batch_size)

                    X6_testi = y_pred[ipat:(ipat+batch_size),itime:(itime+ts),:]
        return y_pred
