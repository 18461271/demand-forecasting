# data processing
import os, sys, time, math, json, random
#import tensorflow as tf
import matplotlib.pyplot as plt
#from keras.utils import np_utils
#%matplotlib inline
import pandas as pd
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
 # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler

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
import sys
from itertools import product, starmap

def storeitems():
    return product(range(1,51), range(1,11))


def storeitems_column_names():
    return list(starmap(lambda i,s: f'item_{i}_store_{s}_sales', storeitems()))


def sales_by_storeitem(df):
    ret = pd.DataFrame(index=df.index.unique())
    for i, s in storeitems():
        #print(i,s)
        #sys.exit()

        ret[f'item_{i}_store_{s}_sales'] = df[(df['item'] == i) & (df['store'] == s)]['sales'].values
        #print( ret)
    return ret

def shift_series(series, days):
    return series.transform(lambda x: x.shift(days))


def shift_series_in_df(df, series_names=[], days_delta=90):
    ret = pd.DataFrame(index=df.index.copy())
    str_sgn = 'future' if np.sign(days_delta) < 0 else 'past'
    for sn in series_names:
        ret[f'{sn}_{str_sgn}_{np.abs(days_delta)}'] = shift_series(df[sn], days_delta)
    return ret


def stack_shifted_sales(df, days_deltas=[1, 90, 360]):
    names = storeitems_column_names()
    dfs = [df.copy()]
    #print(dfs)
    #sys.exit()
    for delta in days_deltas:
        shifted = shift_series_in_df(df, series_names=names, days_delta=delta)
        #print(shifted.shape)
        #sys.exit()
        dfs.append(shifted)
    return pd.concat(dfs, axis=1, sort=False, copy=False)

def generate_data(df_train, df_test ):
    df_train.index = pd.to_datetime(df_train['date'])
    df_train.drop('date', axis=1, inplace=True)
    df_train = sales_by_storeitem(df_train)

    df_test.index = pd.to_datetime(df_test['date'])
    df_test.drop('date', axis=1, inplace=True)
    df_test['sales'] = np.zeros(df_test.shape[0])
    df_test = sales_by_storeitem(df_test)

    col_names = list(zip(df_test.columns, df_train.columns))  #('item_1_store_1_sales', 'item_1_store_1_sales')
    for cn in col_names:
        assert cn[0] == cn[1]

    df_test['is_test'] = np.repeat(True, df_test.shape[0])
    df_train['is_test'] = np.repeat(False, df_train.shape[0])
    df_total = pd.concat([df_train, df_test])
    weekday_df = pd.get_dummies(df_total.index.weekday, prefix='weekday')
    weekday_df.index = df_total.index
    month_df = pd.get_dummies(df_total.index.month, prefix='month')
    month_df.index =  df_total.index
    df_total = pd.concat([weekday_df, month_df, df_total], axis=1)
    assert df_total.isna().any().any() == False
    df_total = stack_shifted_sales(df_total, days_deltas=[1])
    df_total.dropna(inplace=True)
    sales_cols = [col for col in df_total.columns if '_sales' in col and '_sales_' not in col]
    stacked_sales_cols = [col for col in df_total.columns if '_sales_' in col]
    other_cols = [col for col in df_total.columns if col not in set(sales_cols) and col not in set(stacked_sales_cols)]

    sales_cols = sorted(sales_cols)
    stacked_sales_cols = sorted(stacked_sales_cols)

    new_cols = other_cols + stacked_sales_cols + sales_cols
    df_total = df_total.reindex(columns=new_cols)
    assert df_total.isna().any().any() == False
    scaler = MinMaxScaler(feature_range=(0,1))
    cols_to_scale = [col for col in df_total.columns if 'weekday' not in col and 'month' not in col and "is_test" not in col]
    scaled_cols = scaler.fit_transform(df_total[cols_to_scale])
    df_total[cols_to_scale] = scaled_cols
    df_train = df_total[df_total['is_test'] == False].drop('is_test', axis=1)
    df_test = df_total[df_total['is_test'] == True].drop('is_test', axis=1)

    X_cols_stacked = [col for col in df_train.columns if '_past_' in col]
    X_cols_caldata = [col for col in df_train.columns if 'weekday_' in col or 'month_' in col or 'year' in col]
    X_cols = X_cols_stacked + X_cols_caldata

    X = df_train[X_cols]
    y_cols = [col for col in df_train.columns if col not in X_cols]
    y = df_train[y_cols]
    X_test = df_test[X_cols]
    return X, y, X_test, scaler, df_train, df_test, cols_to_scale, y_cols


def ts_data(N,T,D,X):
    X = X.transpose()
    x = np.zeros((N,T,D))
    for n in range(N+19):
        for t in range(T):
            if n<N:
                x[n,t,0] = X.iloc[n][t]
                #None
            else:
                for d in range(1,D):
                    #print(n,t,d, n+d)
                    x[:,t,d] = X.iloc[N+d-1][t]
    return x
