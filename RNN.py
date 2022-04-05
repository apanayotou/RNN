# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:03:53 2022

@author: alexp
"""


import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import preprocessing
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional, RepeatVector, TimeDistributed
from tensorflow.keras import regularizers



def pull_data(year,start,end,min_visits=10, groupby="cat"):
    '''
    Imports the Alphamega database classes used for quering the database.
    Then creates the sql query using the start and end date.
    Finally it pulls the data from the database
    
    Note: This will only work if using a Alphamega VPN.

    Parameters
    ----------
    year : int
        Year you want the data from.
    start : str
        Start of date. Date Format: 'YYYY-MM-DD'
    end : str
        End of date. Date Format: 'YYYY-MM-DD'
    min_visits: int
        Min number of visits per item customer pair.
        
    Returns
    -------
    df : Pandas DataFrame
        Dataframe that has every customer cat pair within the date range greater than the min_visits

    '''

    # Adds path where alpha
    sys.path.insert(1, 'C:/Users/alexp/Desktop/Work/Other/Useful/alphamega_db_classes')
    import alphamega_db_classes as adb

    if year%2 == 0:
        year1 = year
        year2 = year+1
    else:
        year1 = year-1
        year2 = year
    year_str = f'{year1}_{year2}'
    db = adb.insight_sql_querier("Ins_AMe")
    db.open_conn()
    query = (f"""
        SELECT
            Cus_CardNo, 
            Pos_TimeDate , 
            sum([Pos_Quantity]) as qty , 
            [ICa_3DCode] AS  cat
        FROM [Ins_AMe].[dbo].[Ins_PosTransactions{year_str}]
        LEFT JOIN [Ins_Item] ON [Ins_Item].Itm_Code = [Ins_PosTransactions{year_str}].Itm_Code
            WHERE Pos_TimeDate >= '{start}' and  Pos_TimeDate <= '{end}' AND
            concat(ICa_3DCode, Cus_CardNo) IN ( 
                SELECT concat(ICa_3DCode, Cus_CardNo)
                FROM [Ins_AMe].[dbo].[Ins_PosTransactions{year_str}]
				LEFT JOIN [Ins_Item] ON [Ins_Item].Itm_Code = [Ins_PosTransactions{year_str}].Itm_Code
                WHERE Pos_TimeDate >= '{start}' and  Pos_TimeDate <= '{end}'
                GROUP BY ICa_3DCode, Cus_CardNo
                HAVING Cus_CardNo <> '' AND count(distinct Pos_TimeDate) >= 10 
                )
        GROUP BY ICa_3DCode, Cus_CardNo, Pos_TimeDate
        HAVING Cus_CardNo <> '' and  ICa_3DCode<> '969'             
             
        """)

    db.select_db_data(query)
    db.close_conn()
    df = db.get_data().copy()
    return df

def filter_cat_by_amount_cust(df, min_cust=20):
    '''
    Filters out categories where the number of customers is less than
    the min_cust.

    Parameters
    ----------
    df : Pandas Dataframe
        Data frame with cat and customers columns.
    min_cust : TYPE, optional
        The min number of customers who purchased each category. The default is 20.

    Returns
    -------
    df : Pandas Dataframe
        Same Dataframe as input but with only cats with enough customer purchases

    '''
    # groups the dataframe by cat and gets the unique count of other columns
    df_grouped = df.groupby("cat").nunique()
    # filters the group df by the min_cust and save the index of cat ids
    cats_to_keep = df_grouped[df_grouped.Cus_CardNo>min_cust].index
    # filter the main df using the cats_to_keep
    df = df[df.cat.isin(cats_to_keep)]
    return df


def create_gap_df(df):
    '''
    Calculates the gap between each customer's purchase of a category and the previous purchase of that category.
    This is done by sorting the dataframe so every transaction of a given customer and category followed by the previous transaction.
    Then the dataframe is duplicated and the index is offset by -1. 
    The two dataframes are mergered on index, category and customer.
    This gives each row a column with data purchase and previous date purchased.
    Lastly these two columns are subtracted to get the number of days between purchases.
    
    Parameters
    ----------
    df : Pandas Dataframe
        Transactionsal data for eavery=

    Returns
    -------
    new_df : TYPE
        DESCRIPTION.

    '''
    # Sort values to have all transactions of the same customer and category are together and sorted in decending date order.
    df = df.sort_values(["Cus_CardNo","cat","Pos_TimeDate"],ascending=[True,True,False])
    # reset index 
    df = df.reset_index(drop=True)
    # duplicated dataframe
    df_copy = df.copy()
    # subtract 1 from duplicated dataframe
    df_copy.index = df_copy.index-1
    # make index a column in both dfs so we can use it to merge
    df_copy["index_copy"] = df_copy.index
    df["index_copy"] = df.index

    # merging both df and its copy on cat cust and index to make new_df. 
    # new_df has two date colomns now and because index is shifted and df is sorted, 
    # date from old df will have be current date and date from copy will have previous date.
    new_df = df.merge(df_copy,how="inner",right_on=("cat", "Cus_CardNo","index_copy"),left_on=("cat", "Cus_CardNo","index_copy"))

    # Subtrace current with previous date to get gap
    new_df["gap"] = new_df.Pos_TimeDate_x - new_df.Pos_TimeDate_y
    # format to show gap in days.
    new_df.gap = new_df.gap.dt.days

    return new_df


def add_past_time_period(df,t,cols):
    '''
    Adds a column to df. This column contains a list of factors from time period [current_date - t]
    This is done using the same method as add_gap: copy df, shift index by -t, add cols data from that 
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    '''
    
    df["index_copy"] = df.index # add copy of index that will be shifted by -t
    copy = df.copy()
    copy.index_copy = copy.index_copy - t # shifting index copy
    columns = cols + ["cat", "Cus_CardNo","index_copy"]
    copy = copy[columns] # keep only target cols
    copy[t] = list(copy[cols].values)
    copy = copy.drop(columns=cols)
    df = df.merge(copy,how='left',right_on=("cat", "Cus_CardNo","index_copy"),left_on=("cat", "Cus_CardNo","index_copy"))
    return df

def add_past_time_period_dumm(df,t,col,dum_col): # col must be list
    df["index_copy"] = df.index # add copy of index that will be shifted by -t
    copy = df.copy()
    copy.index_copy = copy.index_copy - t # shifting index copy
    copy = copy[[copy,dum_col,"cat", "Cus_CardNo","index_copy"]] # keep only target cols
    dummies = pd.get_dummies(df[dum_col])
    dummies_cols = dummies.columns()
    copy = copy.merge(dummies,left_index=True, right_index=True)
    new_name = col+f'-{t}'
    copy = copy.rename({col:new_name},axis=1)
    df = df.merge(copy,how='left',right_on=("cat", "Cus_CardNo","index_copy"),left_on=("cat", "Cus_CardNo","index_copy"))
    return df

def normalize_col(x):
    mean = x.mean()
    std = x.std()
    norm_x = (x-mean)/std
    return norm_x

def preprocess_df2(df,cat=False,qty=False,scale="standard", scale_y=True):
    cols = ["cat","Cus_CardNo","gap"]
    past_cols = ["gap"]
    if cat:
        cols = cols+["cat_x"]
    if qty:
        cols = cols+["qty_x"]
        past_cols = past_cols + ["qty_x"]
    df = df[cols]
    df["y"] = df.gap
    df.dropna(inplace=True)
    if scale == "MinMax":
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    elif scale == "standard":
        scaler = preprocessing.StandardScaler()
    if scale != "None":
        df.loc[:,past_cols] = scaler.fit_transform(df.loc[:,past_cols])
    if scale_y:
        df["y"] = df.gap
    df.dropna(inplace=True)
    for t in reversed(range (1,using_past+1)):
        df = add_past_time_period(df, t, past_cols)
    if cat:
        df = pd.get_dummies(df,columns=["cat_x"])
    df.dropna(inplace=True)
    df = df.sample(frac = 1)
    return df

def train_test_split(df,frac=0.05):
    gap_df = df.groupby(["cat", "Cus_CardNo"]).agg({"gap": [np.mean, np.median,'count','max']})
    gap_df = gap_df.reset_index()
    test_ic = gap_df[["cat","Cus_CardNo"]].sample(frac=frac)
    train_ic = gap_df[["cat","Cus_CardNo"]].drop(test_ic.index, errors="ignore")
    test = df[(df.cat.isin(test_ic.cat)) & (df.Cus_CardNo.isin(test_ic.Cus_CardNo))]
    train = df[(df.cat.isin(train_ic.cat)) & (df.Cus_CardNo.isin(train_ic.Cus_CardNo))]

    test = test.dropna()
    train = train.dropna()
    

    test_x = test.drop(["cat", "Cus_CardNo",  "gap", "index_copy","y"],axis=1)
    train_x = train.drop(["cat", "Cus_CardNo",  "gap", "index_copy","y"],axis=1)
    if "qty_x" in test.columns:
        test_x = test_x.drop(["qty_x"],axis=1)
        train_x = train_x.drop(["qty_x"],axis=1)
        

    test_y = test["y"].to_numpy()
    train_y = train["y"].to_numpy()

    test_x_array = np.array(test_x.to_numpy().tolist()).astype('float32')
    train_x_array = np.array(train_x.to_numpy().tolist()).astype('float32')

    return test_x_array, test_y, train_x_array, train_y

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
    model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(n_steps, n_features),return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(n_steps, n_features),return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(n_steps, n_features))))
    model.add(Dense(1, activation='linear'))
    opt = tf.keras.optimizers.Adam(lr=0.00001)
    model.compile(loss='mean_squared_error', optimizer= opt, metrics=['mae'])
    return model

def RNN(layers,n_steps,n_features,dropout=0.2,lr=0.00001):
    model = Sequential()
    for i in range(len(layers)):
        nodes = layers[i]
        if i+1 == len(layers):
            model.add(Bidirectional(LSTM(nodes, activation='relu', input_shape=(n_steps, n_features))))
        else:
            model.add(Bidirectional(LSTM(nodes, activation='relu', input_shape=(n_steps, n_features),return_sequences=True)))
        if dropout:
            model.add(Dropout(dropout))

    model.add(Dense(1, activation='linear'))
    opt = tf.keras.optimizers.Adam(lr=lr,clipvalue=1)
    model.compile(loss='mean_squared_error', optimizer= opt, metrics=['mae'])

    return model


def outliers(df,col):
    # Select the first quantile
    q1 = df[col].quantile(.25)

    # Select the third quantile
    q3 = df[col].quantile(.75)

    iqr = q3-q1

    lower_outliers = q1-(iqr*1.5)
    upper_outliers = q3+(iqr*1.5)
    return lower_outliers, upper_outliers

def filter_outlier_custs( df):
    gap_df = df.groupby(["cat", "Cus_CardNo"]).agg({"gap": [np.mean, np.median,'count','max']})
    #gap_df = gap_df.reset_index()
    lower_outliers , upper_outliers = outliers(gap_df["gap"],"max")
    outlier_custs = gap_df[gap_df["gap"]["max"] > upper_outliers].index.get_level_values(1)
    df = df[~df.Cus_CardNo.isin(outlier_custs)]
    return df

def plot_loss_graph(history):
    axes = plt.axes()
    axes.plot(history.history['loss'])
    axes.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def test_model_with_future():
    df_future = pull_data(year=2021, start='2021-06-01', end ='2021-12-30')
    gap_df_future, df_future = create_gap_df(df_future)
    #df_merge_future = df_future.merge(gap_df_future, how="left",left_on=["Itm_Code","Cus_CardNo"],right_on=["Itm_Code","Cus_CardNo"])
    gap_df_future = gap_df_future.reset_index()
    df_future = preprocess_df2(df_future)

    test_x_future = df_future.drop(["Itm_Code", "Cus_CardNo",  "gap", "gap_scale", "index_copy"],axis=1)
    test_y_future = df_future["gap"].to_numpy()

    test_x_future = np.reshape(test_x_future.to_numpy(), (test_x_future.shape[0], test_x_future.shape[1], 1))

    #score = model.evaluate(test_x_future, test_y_future, verbose=0)

    y_pred = model.predict(test_x_future)

    #r2 = r2_score(test_y_future, y_pred.flatten())
    axes = plt.axes()

    axes.scatter(test_y_future,y_pred)

def limit_data(train_x, train_y, test_x, test_y, lim):
    train_x = train_x[:lim]
    train_y = train_y[:lim]
    test_x = test_x[:lim]
    test_y = test_y[:lim]
    return train_x, train_y, test_x, test_y


# fixed paramaters
predict_next = 1
using_past = 25
epochs = 100
batch_size = 64
name = f"predict-next-{predict_next}-using-{using_past}-at-{int(time.time())}"
filter_outlier = False
load_from_db = False
########


# loads data from alphamega database
if load_from_db:
    df = pull_data(year=2021, start='2021-01-01', end ='2021-03-30')
    # removing blank category
    df = df[df.cat!='$  ']
    df.to_csv("cat_data.csv",index=False)

# Loads data from csv
else:
    df = pd.read_csv("cat_data.csv",parse_dates=["Pos_TimeDate"])

df = filter_cat_by_amount_cust(df,20)

# filter for one cat
topCat = df.cat.value_counts().index[1]
#df = df[df.cat==topCat]

df = create_gap_df(df)

df = filter_outlier_custs(df)

df = preprocess_df2(df,cat=False,qty=True,scale="standard",scale_y=True)

test_x_array, test_y, train_x_array, train_y = train_test_split(df)

#model = simple_RNN(using_past,2)

model = RNN([500,500],using_past,2,dropout=None,lr=0.0001)

history = model.fit(train_x_array,
                    train_y,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    validation_data=(test_x_array, test_y)
                    )
plot_loss_graph(history)

y_pred = model.predict(train_x_array)

plt.scatter(train_y,y_pred)
#plt.plot(train_y,train_y,color="red")

print(r2_score(train_y,y_pred))
