# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:18:21 2022

@author: alexp
"""

import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters

from sklearn.metrics import r2_score , mean_absolute_error
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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

def add_past_time_period(df,t,col):
    df["index_copy"] = df.index
    copy = df.copy()
    copy.index_copy = copy.index_copy - t
    copy = copy[[col,"Itm_Code", "Cus_CardNo","index_copy"]]
    new_name = col+f'-{t}'
    copy = copy.rename({col:new_name},axis=1)
    df = df.merge(copy,how='left',right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    return df 

def add_past_time_period_multi(df,t,col): # col must be list
    df["index_copy"] = df.index
    copy = df.copy()
    columns = ["Itm_Code", "Cus_CardNo","index_copy"] + col
    copy.index_copy = copy.index_copy - t
    copy = copy[col]
    new_name = col+f'-{t}'
    copy = copy.rename({col:new_name},axis=1)
    df = df.merge(copy,how='left',right_on=("Itm_Code", "Cus_CardNo","index_copy"),left_on=("Itm_Code", "Cus_CardNo","index_copy"))
    return df 

def preprocess_df2(df,add_cat=False):
    if add_cat:
        df = df[["Itm_Code","Cus_CardNo","gap","cat_x"]]
    else:
        df = df[["Itm_Code","Cus_CardNo","gap"]]    
    #df.loc[:,"gap_scale"] = df.gap.pct_change()
    df.dropna(inplace=True)
    df.loc[:,"gap_scale"] = preprocessing.scale(df.gap.values)
    df.dropna(inplace=True)

    for t in range (1,using_past):
        df = add_past_time_period(df, t, "gap_scale")
    df.dropna(inplace=True)
    if add_cat:
        df = pd.get_dummies(df,columns=['cat_x'])
    df = df.sample(frac = 1)
    return df

def train_test_split(gap_df,df,frac=0.05):
    test_ic = gap_df[["Itm_Code","Cus_CardNo"]].sample(frac=frac)
    train_ic = gap_df[["Itm_Code","Cus_CardNo"]].drop(test_ic.index, errors="ignore")
    test = df[(df.Itm_Code.isin(test_ic.Itm_Code)) & (df.Cus_CardNo.isin(test_ic.Cus_CardNo))]
    train = df[(df.Itm_Code.isin(train_ic.Itm_Code)) & (df.Cus_CardNo.isin(train_ic.Cus_CardNo))]
    return test, train

def hyper_tune(train_x,train_y,test_x_future,test_y_future):
    best_score = 0
    best_est = None
    best_param = None
    n_estimators = [10, 100, 200, 400, 600, 800, 1000]
    max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
    max_leaf_nodes = [None]
    max_features = ["auto"]
    min_samples_leaf  = [1, 2, 4]
    min_samples_split = [2, 5, 10]
    n = 0
    n_params = len(n_estimators) * len(max_depth) * len(max_leaf_nodes) * len(max_features) * len(min_samples_leaf) *len(min_samples_split)
    for n_e in n_estimators:
        for md in max_depth:
            for mln in max_leaf_nodes:
                for mf in max_leaf_nodes:
                    for msl in min_samples_leaf:
                        for mss in min_samples_split:
                            grid = RandomForestRegressor(n_estimators=n_e, max_depth=md)
                            grid.fit(train_x,train_y)
                     
                            r2_future = grid.score(test_x_future,test_y_future)
                            if r2_future > best_score:
                                best_score = r2_future
                                best_est = grid
                                best_param = {"n_estimators":n_e, 
                                              "max_depth":md, 
                                              "max_leaf_nodes":mln, 
                                              "max_features":mf, 
                                              "min_samples_leaf":msl,
                                              "min_samples_split": mss}
                            print(best_score)
                            print(best_param)
                            print(n_params - n)
                            n+=1
    return best_score, best_param, best_est
              

predict_next = 1
using_past = 10
epochs = 3
batch_size = 64
name = f"predict-next-{predict_next}-using-{using_past}-at-{int(time.time())}"

df = pull_data(year=2021, start='2021-01-01', end ='2021-05-30')
df_orig = df.copy

gap_df, df = create_gap_df(df)
df_merge = df.merge(gap_df, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])
gap_df = gap_df.reset_index()

df = preprocess_df2(df)

test , train = train_test_split(gap_df,df)

test_x = test.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
train_x = train.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
test_y = test["gap"].to_numpy()
train_y = train["gap"].to_numpy()


grid = RandomForestRegressor()
grid.fit(train_x,train_y)
y_pred = grid.predict(test_x)
r2, mae = grid.score(test_x,test_y), mean_absolute_error(test_y, y_pred)


#get future data

df_future = pull_data(year=2021, start='2021-06-01', end ='2021-12-30')


gap_df_future, df_future = create_gap_df(df_future)
df_merge_future = df_future.merge(gap_df_future, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])
gap_df_future = gap_df_future.reset_index()

df_future = preprocess_df2(df_future)

test_x_future = df_future.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
test_y_future = df_future["gap"].to_numpy()

test_x_future = grid.predict(test_x_future)
r2_future, mae_future = grid.score(test_x_future,test_y_future), mean_absolute_error(test_y_future, y_pred)



best_score, best_param, best_est = hyper_tune(train_x, train_y, test_x_future, test_y_future)

