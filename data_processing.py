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
import sys
from itertools import product, starmap

warnings.filterwarnings("ignore")
"""print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__))
print("tensorflow version {}".format(tf.__version__))"""

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

def storeitems_column_names(logger_ID):
    #logger_ID = data.fk_network_meter.unique()
    return list(map(lambda id : f"logger_{id}_flow"  , logger_ID)) + list(map(lambda id : f"logger_{id}_pressure"  , logger_ID))



def sales_by_storeitem(df):
    ret = pd.DataFrame(index=df.index.unique())
    logger_ID = df.fk_network_meter.unique()
    for id in logger_ID:
        ret[f"logger_{id}_flow"] = df[(df["fk_network_meter"]==id)]["flow_value_clean"].values
        ret[f"logger_{id}_pressure"] = df[(df["fk_network_meter"]==id)]["pressure_value_clean"].values
    return ret

def shift_series(series, days):
    return series.transform(lambda x: x.shift(days))


def shift_series_in_df(df, series_names=[], days_delta=90):
    ret = pd.DataFrame(index=df.index.copy())
    str_sgn = 'future' if np.sign(days_delta) < 0 else 'past'
    for sn in series_names:
        ret[f'{sn}_{str_sgn}_{np.abs(days_delta)}'] = shift_series(df[sn], days_delta)
    return ret


def stack_shifted_sales(df,logger_ID, days_deltas=[1, 90, 360]):
    names = storeitems_column_names(logger_ID)
    dfs = [df.copy()]
    #print(dfs)
    #sys.exit()
    for delta in days_deltas:
        shifted = shift_series_in_df(df, series_names=names, days_delta=delta)
        #print(shifted.shape)
        #sys.exit()
        dfs.append(shifted)
    return pd.concat(dfs, axis=1, sort=False, copy=False)

def generate_data(data):
    df=data.copy()
    logger_ID = data.fk_network_meter.unique()
    df.index = pd.to_datetime(df['timestamp']).sort_values()
    df.drop('timestamp', axis=1, inplace=True)
    df = sales_by_storeitem(df)


    timeofday=(4*df.index.hour + df.index.minute/15 ).astype(int)
    timeofday_df = pd.get_dummies(timeofday, prefix='timeofday')
    timeofday_df.index = df.index

    weekday_df = pd.get_dummies(df.index.dayofweek, prefix='dayofweek')
    weekday_df.index = df.index

    month_df = pd.get_dummies(df.index.month, prefix='month')
    month_df.index =  df.index

    df_total = pd.concat([timeofday_df, weekday_df, month_df, df], axis=1)
    #assert df_total.isna().any().any() == False
    #sys.exit()
    df_total = stack_shifted_sales(df_total, logger_ID, days_deltas=[1,2,3,4])
    #df_total.dropna(inplace=True)
    sales_cols = [col for col in df_total.columns if '_flow' in col and '_flow_' not in col or  '_pressure'  in col and '_pressure_' not in col]
    #print(sales_cols)
    #sys.exit()
    stacked_sales_cols = [col for col in df_total.columns if '_flow_' in col or '_pressure_' in col]
    other_cols = [col for col in df_total.columns if col not in set(sales_cols) and col not in set(stacked_sales_cols)]

    sales_cols = sorted(sales_cols)
    stacked_sales_cols = sorted(stacked_sales_cols)

    new_cols = other_cols + stacked_sales_cols + sales_cols
    df_total = df_total.reindex(columns=new_cols)
    #assert df_total.isna().any().any() == False

    scaler = MinMaxScaler(feature_range=(0,1))
    cols_to_scale = [col for col in df_total.columns if 'timeofday' not in col and 'dayofweek' not in col and 'month' not in col and "is_test" not in col]
    scaled_cols = scaler.fit_transform(df_total[cols_to_scale])
    df_total[cols_to_scale] = scaled_cols
    #df_train = df_total[df_total['is_test'] == False].drop('is_test', axis=1)

    X_cols_stacked = [col for col in df_total.columns if '_past_' in col]
    X_cols_caldata = [col for col in df_total.columns if 'timeofday_' in col or 'dayofweek_' in col or 'month_' in col or 'year' in col]
    X_cols_pressure = [col for col in df_total.columns if  '_pressure'  in col and '_pressure_' not in col]
    X_cols = X_cols_stacked + X_cols_caldata  + X_cols_pressure

    X = df_total[X_cols]
    y_cols = [col for col in df_total.columns if col not in X_cols]
    y = df_total[y_cols]

    return X, y,  scaler,  cols_to_scale, y_cols


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
