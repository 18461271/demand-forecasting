# data processing
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

warnings.filterwarnings("ignore")
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__))
print("tensorflow version {}".format(tf.__version__))

def set_seed(sd=123): #define seed for consistent output
    from numpy.random import seed
    from tensorflow import set_random_seed
    import random as rn
    ## numpy random seed
    seed(sd)
    ## core python's random number
    rn.seed(sd)
    ## tensor flow's random number
    set_random_seed(sd)
#set_seed(sd=123)

def expand_df(df):
    data = df.copy()
    data['day'] = data.index.day
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofweek'] = data.index.dayofweek
    return data


def generate_data(data):
    data = expand_df(data)
    items_id = data.groupby('item')['sales'].apply(list).index
    items_sales = data.groupby('item')['sales'].apply(list)#.series

    stores_id  = data.groupby('store')['item'].apply(list).index
    items_stores = data.groupby('item')['store'].apply(list)

    items_day = data.groupby('item')['day'].apply(list)
    items_month = data.groupby('item')['month'].apply(list)
    items_dayofweek = data.groupby('item')['dayofweek'].apply(list)

    item_len = len(items_id)
    store_len = len(stores_id)
    day_len = 31
    month_len = 12
    week_len = 7


    N = item_len #50
    print(N)
    #sys.exit()
    T = len( items_day[1])  #int( len( items_day[1]) / len(stores_id) ) #18620
    D = 1

    X1 = np.zeros((N,T,1))
    X2 = np.zeros((N,T,1))
    X3 = np.zeros((N,T,1))
    X4 = np.zeros((N,T,1))
    X5 = np.zeros((N,T,1))

    y = np.zeros((N,T,D))

    for i,sku in enumerate(items_id):
        day = items_day[sku]
        month = items_month[sku]
        dayofweek = items_dayofweek[sku]
        stores = items_stores[sku]#np.array(items_stores[sku])
        #print(stores)
        #stores = stores.reshape( (18260,1))
        sales = items_sales[sku]
        #print(type(day),len(sales))
        #sys.exit()

        for t in range(T):
            X1[i,t] = str(sku-1)
            X2[i,t] = str(stores[t]-1)
            X3[i,t] = str(day[t]-1)
            X4[i,t] = str(month[t]-1)
            X5[i,t] = str(dayofweek[t])
            y[i,t] = sales[t]
            #print(str(X2[i,t]), type(str(X2[i,t])))
            #print(str(X3[i,t]), type(str(X3[i,t])))
            #print(str(X4[i,t]), type(str(X4[i,t])))
            #print(str(X5[i,t]), type(str(X5[i,t])))
            #print((y[i,t]), type((y[i,t])))
            #sys.exit()

    y_avg = np.mean(y, axis = 1)
    y = np.array([y[i]/y_avg[i] for i in range(y.shape[0]) ]) #scale sales data by dividing the average sale of each item
    prop_train = 0.80
    ntrain = int(X1.shape[1]*prop_train)+2

    X1_train, X1_val = X1[:,:ntrain], X1[:,ntrain:]
    X2_train, X2_val = X2[:,:ntrain], X2[:,ntrain:]
    X3_train, X3_val = X3[:,:ntrain], X3[:,ntrain:]
    X4_train, X4_val = X4[:,:ntrain], X4[:,ntrain:]
    X5_train, X5_val = X5[:,:ntrain], X5[:,ntrain:]
    y_train, y_val = y[:,:ntrain], y[:,ntrain:]
    X6_train, X6_val = np.zeros_like(y_train), np.zeros_like(y_val) # y[:,:ntrain-1],  y[:,ntrain-1:-1]
    X6_train[:,0] = np.mean(y, axis = 1)
    X6_train[:,1:] = y[:,:ntrain-1]
    X6_val[:] = y[:,ntrain-1:-1]
    X6 = np.hstack((X6_train, X6_val ))
    return (X1,X2, X3, X4, X5, X6, y ), (X1_train,X2_train,X3_train,X4_train,X5_train,X6_train, y_train ),(X1_val,X2_val,X3_val,X4_val,X5_val,X6_val,y_val),y_avg

def generate_test_data(data):
    #y = np.zeros((N,T,D))
    data = expand_df(data)
    items_id = data.groupby('item')['store'].apply(list).index
    #items_sales = data.groupby('item')['sales'].apply(list)#.series

    stores_id  = data.groupby('store')['item'].apply(list).index

    items_stores = data.groupby('item')['store'].apply(list)
    items_day = data.groupby('item')['day'].apply(list)
    items_month = data.groupby('item')['month'].apply(list)
    items_dayofweek = data.groupby('item')['dayofweek'].apply(list)


    item_len = len(items_id)
    store_len = len(stores_id)
    day_len = 31
    month_len = 12
    week_len = 7

    N = item_len
    T = len( items_day[1])
    D = 1


    X1 = np.zeros((N,T,1))
    X2 = np.zeros((N,T,1))
    X3 = np.zeros((N,T,1))
    X4 = np.zeros((N,T,1))
    X5 = np.zeros((N,T,1))

    for i,sku in enumerate(items_id):

        stores = items_stores[sku]
        day = items_day[sku]
        month = items_month[sku]
        dayofweek = items_dayofweek[sku]
        for t in range(T):
            #X1[i,t] = sku-1
            #X2[i,t] = stores[t]-1
            #X3[i,t] = day[t]-1
            #X4[i,t] = month[t]-1
            #X5[i,t] = dayofweek[t]

            X1[i,t] = str(sku-1)
            X2[i,t] = str(stores[t]-1)
            X3[i,t] = str(day[t]-1)
            X4[i,t] = str(month[t]-1)
            X5[i,t] = str(dayofweek[t])
            #y[i,t] = sales[t]
    #(X1_val,X2_val,X3_val,X4_val,X5_val,X6_val,y_val)

    return (X1, X2, X3, X4, X5 )
