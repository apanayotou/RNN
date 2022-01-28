# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:03:53 2022

@author: alexp
"""


import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score , mean_absolute_error
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import r2_score


# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/alexp/Desktop/Work/Other/Useful/alphamega_db_classes')

import alphamega_db_classes as adb

def pull_data(year,start,end):
    if year%2 == 0:
        year1 = year
        year2 = year+1
    else:
        year1 = year-1
        year2 = year
    year_str = f'{year1}_{year2}'
    db = adb.insight_sql_querier("Ins_AMe")
    db.open_conn()
    query = (
        'SELECT  \n'
            f'[Ins_PosTransactions{year_str}].Itm_Code, Cus_CardNo, Pos_TimeDate , sum([Pos_Quantity]) as qty , MAX([ICa_3DCode]) AS  cat \n'
        f'FROM [Ins_AMe].[dbo].[Ins_PosTransactions{year_str}] \n'
        f'LEFT JOIN [Ins_Item] ON [Ins_Item].Itm_Code = [Ins_PosTransactions{year_str}].Itm_Code \n'
            f"WHERE Pos_TimeDate >= '{start}' and  Pos_TimeDate <= '{end}' and \n"
            f'concat([Ins_PosTransactions{year_str}].Itm_Code, Cus_CardNo) IN ( \n'
                'SELECT concat(Itm_Code, Cus_CardNo) \n'
                f'FROM [Ins_AMe].[dbo].[Ins_PosTransactions{year_str}] \n'
                f"WHERE Pos_TimeDate >= '{start}' and  Pos_TimeDate <= '{end}' \n"
                'GROUP BY Itm_Code, Cus_CardNo \n'
                "HAVING Cus_CardNo <> '' AND count( distinct Pos_TimeDate) > 20 "
                ") \n"
        f'GROUP BY [Ins_PosTransactions{year_str}].Itm_Code, Cus_CardNo, Pos_TimeDate \n'
        "HAVING Cus_CardNo <> '' "
        )
    db.select_db_data(query)
    db.close_conn()
    df = db.get_data().copy()    
    return df

def create_gap_df(df):
    df = df.sort_values(["Cus_CardNo","Itm_Code","Pos_TimeDate"],ascending=[True,True,False])
    df = df.reset_index(drop=True)
    df_copy = df.copy()
    df_copy.index = df_copy.index-1
    df_copy["index_copy"] = df_copy.index
    df["index_copy"] = df.index
    new_df = df.merge(df_copy,how="inner",right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    new_df["gap"] = new_df.Pos_TimeDate_x - new_df.Pos_TimeDate_y
    new_df.gap = new_df.gap.dt.days
    gap_df = new_df.groupby(["Itm_Code", "Cus_CardNo"]).agg({"gap": [np.mean, np.median,'count'],"Pos_TimeDate_x":["max"]})
    return gap_df, new_df


def preprocess_df(df):
    #36.55 sec for 10000
    df = df[["Itm_Code","Cus_CardNo","gap"]]
    X = []
    y = [] 
    df.loc[:,"gap_scale"] = df.gap.pct_change()
    df.dropna(inplace=True)
    df.gap_scale = preprocessing.scale(df.gap_scale.values)
    df.dropna(inplace=True)
    n=0
    for i in df.index: 
        print(n-len(df))
        itm, cust, gap = df.loc[i,["Itm_Code","Cus_CardNo","gap"]]
        array = df[(df.Itm_Code==itm) & (df.Cus_CardNo==cust)]

        array = array.loc[i+predict_next:i+using_past,"gap_scale"].to_list()
        
        if len(array)<10:
            pass
        else:
            X.append(array)
            y.append(gap)
        n+=1
    
    return X,y

def add_past_time_period(df,t,cols):
    df["index_copy"] = df.index # add copy of index that will be shifted by -t
    copy = df.copy()
    copy.index_copy = copy.index_copy - t # shifting index copy
    columns = cols + ["Itm_Code", "Cus_CardNo","index_copy"]
    copy = copy[columns] # keep only target cols 
    copy[i] = list(copy[col].values)
    df = df.merge(copy,how='left',right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    return df 

def add_past_time_period_dumm(df,t,col,dum_col): # col must be list
    df["index_copy"] = df.index # add copy of index that will be shifted by -t
    copy = df.copy()
    copy.index_copy = copy.index_copy - t # shifting index copy
    copy = copy[[copy,dum_col,"Itm_Code", "Cus_CardNo","index_copy"]] # keep only target cols 
    dummies = pd.get_dummies(df[dum_col])
    dummies_cols = dummies.columns()
    copy = copy.merge(dummies,left_index=True, right_index=True)
    new_name = col+f'-{t}'
    copy = copy.rename({col:new_name},axis=1)
    df = df.merge(copy,how='left',right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    return df 


def preprocess_df2(df,cat=False,qty=False):
    cols = ["Itm_Code","Cus_CardNo","gap"]
    if cat:
        cols = cols+["cat_x"]
    if qty:
        cols = cols+[""]
    df = df[cols]
    #df.loc[:,"gap_scale"] = df.gap.pct_change()
    df.dropna(inplace=True)
    df.loc[:,"gap_scale"] = preprocessing.scale(df.gap.values)
    df.dropna(inplace=True)
    for t in reversed(range (1,using_past+1)):
        df = add_past_time_period(df, t, "gap_scale")
    df = pd.get_dummies(df,columns=["cat_x"])
    df.dropna(inplace=True)
    df = df.sample(frac = 1)
    return df

def train_test_split(gap_df,df,frac=0.05):
    test_ic = gap_df[["Itm_Code","Cus_CardNo"]].sample(frac=frac)
    train_ic = gap_df[["Itm_Code","Cus_CardNo"]].drop(test_ic.index, errors="ignore")
    test = df[(df.Itm_Code.isin(test_ic.Itm_Code)) & (df.Cus_CardNo.isin(test_ic.Cus_CardNo))]
    train = df[(df.Itm_Code.isin(train_ic.Itm_Code)) & (df.Cus_CardNo.isin(train_ic.Cus_CardNo))]
    return test, train

def encoder_decoder(n_steps_in,n_features,n_steps_out):
    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features))))
    model.add(RepeatVector(n_steps_out))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse',metrics=['mae'])
    return model

def simple_RNN(n_steps,n_features):
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(n_steps, n_features))))

    model.add(Dense(1, activation='linear'))
    #model.compile(optimizer='adam', loss='mse', metric="mae")
    opt = tf.keras.optimizers.Adam(lr=0.00001,clipvalue=,epsilon=1e-04)
    model.compile(loss='mean_squared_error', optimizer= opt, metrics=['mae'])
    return model

predict_next = 1
using_past = 15
epochs = 500
batch_size = 64
name = f"predict-next-{predict_next}-using-{using_past}-at-{int(time.time())}"

df = pull_data(year=2021, start='2021-01-01', end ='2021-05-30')
df_orig = df.copy

gap_df, df = create_gap_df(df)
df_merge = df.merge(gap_df, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])
gap_df = gap_df.reset_index()

df = preprocess_df2(df)

test , train = train_test_split(gap_df,df)

test = test.dropna()
train = train.dropna()
test_x = test.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
train_x = train.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
test_y = test["gap"].to_numpy()
train_y = train["gap"].to_numpy()

lim = 10000
train_x = train_x[:lim]
train_y = train_y[:lim]
test_x = test_x[:lim]
test_y = test_y[:lim]
test_x_array = np.reshape(test_x.to_numpy(), (test_x.shape[0], test_x.shape[1], 1))
train_x_array = np.reshape(train_x.to_numpy(), (train_x.shape[0], train_x.shape[1], 1))

model = simple_RNN(test_x.shape[1],1)

tenserboard = TensorBoard(log_dir=f'logs/{name}')

filepath = "RNN_Final-{epoch:02d}-{mae:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones





# Save model

'''
df_future = pull_data(year=2021, start='2021-06-01', end ='2021-12-30')


gap_df_future, df_future = create_gap_df(df_future)
df_merge_future = df_future.merge(gap_df_future, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])
gap_df_future = gap_df_future.reset_index()

df_future = preprocess_df2(df_future)

test_x_future = df_future.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
test_y_future = df_future["gap"].to_numpy()

test_x_future = np.reshape(test_x_future.to_numpy(), (test_x_future.shape[0], test_x_future.shape[1], 1))

score = model.evaluate(test_x_future, test_y_future, verbose=0)


y_pred = model.predict(test_x_future)

r2 = r2_score(test_y_future, y_pred.flatten())
axes = plt.axes()

axes.scatter(test_y_future,y_pred)
'''

history = model.fit(train_x_array, 
                    train_y, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=1,
                    validation_data=(test_x_array, test_y)
                    )
               
                    