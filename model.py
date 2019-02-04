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


def define_model(time_steps,
                    item_len,
                    store_len,
                    day_len,
                    month_len,
                    week_len,
                    hidden_neurons = 50,
                    batch_size=None,
                    stateful=False):
    output_neurons = 1
    input_1 = Input(batch_shape=(batch_size,time_steps,1))#item 50
    input_01 = TimeDistributed(Embedding(item_len, 10, input_length=1))(input_1)
    input_01 = TimeDistributed(Flatten())(input_01)

    input_2 = Input(batch_shape=(batch_size,time_steps,1))#store 10
    input_02 = TimeDistributed(Embedding(store_len, 4, input_length=1))(input_2)
    input_02 = TimeDistributed(Flatten())(input_02)

    input_3 = Input(batch_shape=(batch_size,time_steps,1))#day 31
    input_03 = TimeDistributed(Embedding(day_len, 8, input_length=1))(input_3)
    input_03 = TimeDistributed(Flatten())(input_03)

    input_4 = Input(batch_shape=(batch_size,time_steps,1))#month 12
    input_04 = TimeDistributed(Embedding(month_len, 4, input_length=1))(input_4)
    input_04 = TimeDistributed(Flatten())(input_04)

    input_5 = Input(batch_shape=(batch_size,time_steps,1))#week 7
    input_05 = TimeDistributed(Embedding(week_len, 4, input_length=1))(input_5)
    input_05 = TimeDistributed(Flatten())(input_05)

    input_6 = Input(batch_shape=(batch_size,time_steps,1))#previous sales 7
    input_06 = Reshape(target_shape=(time_steps, 1,))(input_6)
    input_06 = TimeDistributed(Dense(8))(input_06)

    features = concatenate([input_01, input_02, input_03, input_04, input_05, input_06])
    features = Dense(100)(features) #TimeDistributed(Dense(64))(features)
    features = Dropout(0.4)(features)

    lstm  = LSTM( hidden_neurons,activation='sigmoid',kernel_initializer='normal', dropout=0.25, return_sequences=True, stateful=stateful, name="lstm")(features)

    target_model = TimeDistributed(Dense(output_neurons,name="dense"))(lstm)
    model = models.Model(inputs=[input_1, input_2, input_3, input_4, input_5, input_6], outputs = target_model )

    model.compile(loss="mean_squared_error",sample_weight_mode="temporal",optimizer="rmsprop")

    return (model,(features,lstm,target_model))
#https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
class statefulModel(object):
    def __init__(self,model,print_val_every = 500):
        '''
        model must be stateful keras model object
        batch_input_shape must be specified
        '''
        bis = model.layers[0].get_config()["batch_input_shape"]
        print("batch_input_shape={}".format(bis))
        self.batch_size = bis[0]
        self.ts         = bis[1]
        self.Nfeat      = bis[2]
        self.model      = model
        self.print_val_every = print_val_every

    def train1epoch(self, X, y, epoch=None):
        '''
        devide the training set of time series into batches.
        '''
        print( "  Training..")
        [X1,X2,X3,X4,X5,X6 ] = X
        batch_index = np.arange(X1.shape[0])
        ## shuffle to create batch containing different time series
        np.random.shuffle(batch_index)
        count = 1

        #print("batch_index",batch_index)
        for ibatch in range(self.batch_size,X1.shape[0]+1, self.batch_size):

            print( "    Batch {:02d}".format(count))
            #print( "ibatch",ibatch )
            pick = batch_index[(ibatch-self.batch_size):ibatch]
            #print("pick", pick  , len(pick))
            if len(pick) < self.batch_size:
                continue
            X1_batch = X1[pick]
            X2_batch = X2[pick]
            X3_batch = X3[pick]
            X4_batch = X4[pick]
            X5_batch = X5[pick]
            X6_batch = X6[pick]
            y_batch = y[pick]
            print("count", count)
            print(X1_batch.shape)
            #ibatch = 50

            X_batch = [X1_batch,X2_batch,X3_batch,X4_batch,X5_batch,X6_batch]
            self.fit_across_time(X_batch,y_batch,epoch,ibatch)
            count += 1
            print("count", count)

    def fit_across_time(self, X, y, epoch=None, ibatch=None):
        '''
        training for the given set of time series
        It always starts at the time point 0 so we need to reset states to zero.
        '''
        self.model.reset_states()
        [X1,X2,X3,X4,X5,X6] = X
        for itime in range(self.ts,X1.shape[1]+1,self.ts):
            ## extract sub time series
            print("itime",itime)
            X1time = X1[:,itime-self.ts:itime,:]
            X2time = X2[:,itime-self.ts:itime,:]
            X3time = X3[:,itime-self.ts:itime,:]
            X4time = X4[:,itime-self.ts:itime,:]
            X5time = X5[:,itime-self.ts:itime,:]
            X6time = X6[:,itime-self.ts:itime,:]
            ytime = y[:,itime-self.ts:itime,:]

            Xtime = [X1time,X2time,X3time,X4time,X5time,X6time]
            #print(X1.shape, X1time.shape ,ytime.shape )

            val = self.model.fit(Xtime,ytime,
                        nb_epoch=1,
                        ## no shuffling across rows (i.e. time series)
                        shuffle=False,
                        ## use all the samples in one epoch
                        batch_size=X1time.shape[0],
                        verbose= False)
            if itime % self.print_val_every == 0:
                print( "      {start:4d}:{end:4d} loss={val:.3f}".format(
                start=itime-self.ts, end=itime, val=val.history["loss"][0]))
                sys.stdout.flush()
                ## uncomment below if you do not want to save weights for every epoch every batch and every time
        if epoch is not None:
            #path = "../output" ../output/

            self.model.save_weights(
                "weights_epoch{:03d}_batch{:01d}.hdf5".format(epoch+1, ibatch))

    def fit(self, X, y, epochs=300):

        [ X1, X2 ,X3, X4, X5, X6] = X

        past_val_loss = np.Inf
        history = []
        for iepoch in range(epochs):
            self.model.reset_states()
            print( "__________________________________")
            print( "Epoch {}".format(iepoch+1))

            self.train1epoch( X, y, iepoch)
        return history
