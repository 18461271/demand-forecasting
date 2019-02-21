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
from keras.layers import LSTM, Dense, Conv1D, Input, Dropout, AvgPool1D, Reshape,MaxPooling2D
from keras.layers.embeddings import Embedding


from keras.layers import RepeatVector

from keras.layers.convolutional import MaxPooling1D
#import seaborn as sns
import warnings
from keras import models, layers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#once off fit
def define_model_1(time_steps,
                    hidden_neurons = 50,
                    batch_size=None,
                    feature_dim =20,
                    stateful=False):
    output_neurons = 1

    inputs =Input(shape=(time_steps, feature_dim)) #Input(batch_shape=(batch_size,time_steps, feature_dim))  #Input(shape=(X_train.shape[1], X_train.shape[2]))

    lstm  = LSTM( hidden_neurons,activation='sigmoid',kernel_initializer='normal', dropout=0.25, return_sequences=True, stateful=stateful, name="lstm")(inputs)

    target_model = TimeDistributed(Dense(output_neurons,name="dense"))(lstm)
    model = models.Model(inputs=inputs, outputs = target_model )

    model.compile(loss="mean_squared_error",optimizer="rmsprop")

    return (model,(inputs,lstm,target_model))

#----------------------------------------------------------------------------
#fit accross time
def define_model_2(time_steps,
                    hidden_neurons = 50,
                    batch_size=None,
                    feature_dim =20,
                    stateful=False):
    output_neurons = 1

    inputs =Input(batch_shape=(batch_size,time_steps, feature_dim))  #Input(shape=(X_train.shape[1], X_train.shape[2]))

    lstm  = LSTM( hidden_neurons,activation='sigmoid',kernel_initializer='normal',dropout=0.25, return_sequences=True, stateful=stateful, name="lstm")(inputs)


    #lstm2 = LSTM( hidden_neurons, activation='sigmoid',return_sequences=True, stateful=stateful)(lstm)
    target_model = TimeDistributed(Dense(output_neurons,name="dense"))(lstm)
    model = models.Model(inputs=inputs, outputs = target_model )

    model.compile(loss="mean_squared_error",optimizer="rmsprop")

    return (model,(inputs,lstm,target_model))
#----------------------------------------------------------------------------
def define_model_3(time_steps,
                    hidden_neurons = 50,
                    batch_size=None,
                    feature_dim =20,
                    stateful=False):
    output_neurons = 1

    inputs = Input(shape=(time_steps, feature_dim))  #Input(shape=(X_train.shape[1], X_train.shape[2]))
    conv1 = Conv1D(filters=32,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(inputs)
    lstm1 = LSTM(hidden_neurons, kernel_initializer ='normal',dropout=0.25, return_sequences = True, stateful=stateful, name="lstm")(conv1) #activation ='elu',

    target_model = TimeDistributed(Dense(output_neurons, name = "dense"))(lstm1)
    model = models.Model(inputs=inputs, outputs = target_model )

    model.compile(loss="mean_squared_error",sample_weight_mode = "temporal",optimizer = "adam")

    return model,target_model
#https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html

#----------------------------------------------------------------------------
def define_model_4(time_steps,
                    hidden_neurons = 50,
                    batch_size=None,
                    feature_dim =20,
                    stateful=False):
    output_neurons = 1

    inputs = Input(batch_shape=(batch_size,time_steps, feature_dim)) #Input(shape=(X_train.shape[1], X_train.shape[2]))
    conv1 = Conv1D(filters=32,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(inputs)
    conv2 = Conv1D(filters=32,
                   kernel_size=8,
                   activation='relu',
                   padding='same')(conv1)

    lstm1 = LSTM(hidden_neurons, activation ='sigmoid',kernel_initializer ='normal',dropout=0.25, return_sequences = True, stateful=stateful, name="lstm")(conv2) #activation ='elu',

    target_model = TimeDistributed(Dense(output_neurons, name = "dense"))(lstm1)
    model = models.Model(inputs=inputs, outputs = target_model )

    model.compile(loss="mean_squared_error",sample_weight_mode = "temporal",optimizer = "adam")

    return model,target_model
#https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
#----------------------------------------------------------------------------

def define_model_6(time_steps,
                    hidden_neurons = 50,
                    batch_size=None,
                    feature_dim =20,
                    stateful=False):
    output_neurons = 1

    inputs = Input(batch_shape=(batch_size,time_steps, feature_dim)) #Input(shape=(time_steps, feature_dim))  #Input(shape=(X_train.shape[1], X_train.shape[2]))
    conv1 = Conv1D(filters=32,
                   kernel_size=1,
                   activation='relu',
                   padding='same'
                   )(inputs)

    conv2 = Conv1D(filters=32,
                   kernel_size=8,
                   activation='relu',
                   padding='same')(conv1)


    pool = MaxPooling1D(pool_size=2,
                    padding='same')(conv2 )


    lstm1 = LSTM(hidden_neurons, return_sequences=True)(pool)
    print("here4")
    dense = TimeDistributed(Dense(20,name="dense", activation='relu'))(lstm1)
    print("here5")

    target_model = TimeDistributed(Dense(output_neurons,name="dense"))(dense)
    print("here6")

    model = models.Model(inputs=inputs, outputs = target_model )

    model.compile(loss="mean_squared_error",optimizer="adam")
    print("here7")
    return model,target_model

#----------------------------------------------------------------------------
def define_model_5(time_steps,
                    hidden_neurons = 50,
                    batch_size=None,
                    feature_dim =20,
                    stateful=False):
    output_neurons = 1

    inputs = Input(batch_shape=(batch_size,time_steps, feature_dim)) #Input(shape=(time_steps, feature_dim))  #Input(shape=(X_train.shape[1], X_train.shape[2]))

    top_lstm = LSTM(hidden_neurons, return_sequences=True)(inputs)
    top_dense = TimeDistributed(Dense(hidden_neurons, activation='relu'))(top_lstm)
    top_dropout =TimeDistributed( Dropout(0.5))(top_dense)
    print(top_dropout.shape )
    # bottom pipeline
    #bottom_dense = Dense(hidden_neurons)(inputs)
    bottom_conv1 = Conv1D(filters=32,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(inputs)
    print(bottom_conv1.shape )
    """bottom_pooling = AvgPool1D(
                    pool_size=60,
                    padding='same')(bottom_conv1)
    print("bottom_pooling",bottom_pooling.shape )
    bottom_reshape = Reshape(
                    target_shape=[32])(bottom_pooling)"""
    # concat output from both pipelines
    final_concat = Concatenate()([top_dropout, bottom_conv1])
    final_dense =  TimeDistributed(Dense(output_neurons)(final_concat))

    model = models.Model(inputs=inputs, outputs = final_dense )

    model.compile(loss="mean_squared_error",sample_weight_mode="temporal",optimizer="rmsprop")

    return model,target_model
#----------------------------------------------------------------------------
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
        batch_index = np.arange(X.shape[0])
        ## shuffle to create batch containing different time series
        np.random.shuffle(batch_index)
        count = 1

        #print("batch_index",batch_index)
        for ibatch in range(self.batch_size,X.shape[0]+1, self.batch_size):

            #print( "    Batch {:02d}".format(count))
            #print( "ibatch",ibatch )
            pick = batch_index[(ibatch-self.batch_size):ibatch]
            #print("pick", pick  , len(pick))
            if len(pick) < self.batch_size:
                continue

            X_batch = X[pick]
            y_batch = y[pick]
            #print("count", count)
            #print(X1_batch.shape)
            #ibatch = 50

            self.fit_across_time(X_batch,y_batch,epoch,ibatch)
            count += 1
            #print("count", count)

    def fit_across_time(self, X, y, epoch=None, ibatch=None):
        '''
        training for the given set of time series
        It always starts at the time point 0 so we need to reset states to zero.
        '''
        self.model.reset_states()

        for itime in range(self.ts,X.shape[1]+1,self.ts):
            ## extract sub time series
            #print("itime",itime)

            Xtime = X[:,itime-self.ts:itime,:]
            ytime = y[:,itime-self.ts:itime,:]

            #print(X1.shape, X1time.shape ,ytime.shape )

            val = self.model.fit(Xtime,ytime,
                        nb_epoch=1,
                        ## no shuffling across rows (i.e. time series)
                        shuffle=False,
                        ## use all the samples in one epoch
                        batch_size=Xtime.shape[0],
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

        past_val_loss = np.Inf
        history = []
        for iepoch in range(epochs):
            self.model.reset_states()
            print( "__________________________________")
            print( "Epoch {}".format(iepoch+1))

            self.train1epoch( X, y, iepoch)

        #self.model.save_weights("weights_epoch011_batch500.hdf5")


        return history
